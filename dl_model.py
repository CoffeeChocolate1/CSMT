

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, GlobalAveragePooling3D
import numpy as np

# ---------- 激活 ----------
class SWiGLU(tf.keras.layers.Layer):
    def call(self, inputs):
        x, gate = tf.split(inputs, 2, axis=-1)
        return x * tf.sigmoid(gate) + x

# ---------- 位置编码 ----------
def Positional_Encoding(max_len, d_emb):
    pos = np.arange(max_len)[:, np.newaxis]
    i   = np.arange(d_emb)[np.newaxis, :]
    angle = pos / np.power(10000, 2 * i / d_emb)
    PE = np.zeros((max_len, d_emb))
    PE[:, 0::2] = np.sin(angle[:, 0::2])
    PE[:, 1::2] = np.cos(angle[:, 1::2])
    return PE[np.newaxis, ...]

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def call(self, x):
        # x shape: (B, L, D)
        seq_len = tf.shape(x)[1]
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        dim = tf.range(self.embed_dim, dtype=tf.float32)[tf.newaxis, :]
        angle = pos / tf.pow(10000.0, 2 * dim / self.embed_dim)
        sin_mask = tf.cast(dim % 2 == 0, tf.float32)
        pos_enc = sin_mask * tf.sin(angle) + (1 - sin_mask) * tf.cos(angle)
        return x + pos_enc[tf.newaxis, :, :]

    def get_config(self):
        return {"embed_dim": self.embed_dim}


class DMHSA(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads, self.d_model, self.depth = num_heads, d_model, d_model // num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)
        self.dropout = Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def differential_attention(self, scores):
        # scores shape: (B, H, L, L)
        diff = scores[:, :, 1:, :] - scores[:, :, :-1, :]   # difference along query positions
        diff = tf.pad(diff, [[0,0],[0,0],[1,0],[0,0]])
        return diff

    def call(self, q, k, v, training):
        bs = tf.shape(q)[0]
        q, k, v = [self.split_heads(self.wq(q), bs),
                   self.split_heads(self.wk(k), bs),
                   self.split_heads(self.wv(v), bs)]
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        # differential enhancement (keeps shape compatibility)
        scores = scores + self.differential_attention(scores)
        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, training=training)
        out = tf.matmul(attn, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (bs, -1, self.d_model))
        return self.dense(out), attn      # return attention as well

# ---------- Transformer Block ----------
class SST(tf.keras.layers.Layer):
    def __init__(self, num_heads, transformer_layer_depth, d_model, dff,
                 dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.depth = transformer_layer_depth
        self.mha = DMHSA(num_heads, d_model, dropout_rate)
        self.drop1 = Dropout(dropout_rate)
        self.ln1   = LayerNormalization(epsilon=layernorm_eps)
        self.dense1 = Dense(dff)
        self.swiglu = SWiGLU()
        self.dense2 = Dense(d_model)
        self.drop2 = Dropout(dropout_rate)
        self.ln2   = LayerNormalization(epsilon=layernorm_eps)

    def call(self, inputs, training):
        x = inputs
        attn_weights = None
        for _ in range(self.depth):
            attn_out, attn_weights = self.mha(x, x, x, training=training)
            attn_out = self.drop1(attn_out, training=training)
            x = self.ln1(x + attn_out)
            ffn = self.dense2(self.drop2(self.swiglu(self.dense1(x)), training=training))
            x = self.ln2(x + ffn)
        return x, attn_weights


class ChannelAttention3D(tf.keras.layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.pool = GlobalAveragePooling3D()
        # we'll build FC layers lazily when input shape known

    def build(self, input_shape):
        # input_shape: (B, D, H, W, C)
        _, _, _, _, C = input_shape
        reduced = max(4, C // self.reduction)
        self.fc1 = Dense(reduced, activation='relu', name='ca_fc1')
        self.fc2 = Dense(C, activation='sigmoid', name='ca_fc2')
        super().build(input_shape)

    def call(self, x):
        # x: (B, D, H, W, C)
        w = self.pool(x)               # (B, C)
        w = self.fc1(w)
        w = self.fc2(w)                # (B, C)
        w = tf.reshape(w, (-1, 1, 1, 1, tf.shape(w)[-1]))  # (B,1,1,1,C)
        return x * w


class SpatialAttention3D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        # conv will be built lazily

    def build(self, input_shape):
        # conv3d with output channel 1
        self.conv = layers.Conv3D(1, kernel_size=(1, self.kernel_size, self.kernel_size),
                                  padding='same', activation=None, name='sa_conv3d')
        super().build(input_shape)

    def call(self, x):
        # x: (B, D, H, W, C)
        # channel-wise pooling
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)   # (B,D,H,W,1)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)    # (B,D,H,W,1)
        cat = tf.concat([avg_pool, max_pool], axis=-1)         # (B,D,H,W,2)
        # apply conv (we use Conv3D with kernel (1,k,k) to focus spatial dims)
        attn = self.conv(cat)                                  # (B,D,H,W,1)
        attn = tf.nn.sigmoid(attn)
        return x * attn


def CSMT(WS, k, Num_Classes, num_transformer_layers=2, num_heads=4,
         dff=4*64, dropout_rate=0.1, layernorm_eps=1e-3):
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    input_shape = (WS, WS, k)
    patch_size = 3
    embed_dim = 64

    inputs = layers.Input(shape=input_shape, name='inputs')

    # Expand dims to have a depth dimension for Conv3D: (B, 1, WS, WS, k)
    x = tf.expand_dims(inputs, axis=1)

    # Patch embedding (Conv3D)
    patch_embed = layers.Conv3D(embed_dim,
                                kernel_size=(1, patch_size, patch_size),
                                strides=(1, patch_size, patch_size),
                                padding='valid',
                                name='patch_embed')(x)   # (B, 1, H_out, W_out, embed_dim)



    ca = ChannelAttention3D(reduction=8, name='channel_attention3d')(patch_embed)

    sa = SpatialAttention3D(kernel_size=3, name='spatial_attention3d')(ca)

    refined = sa  # (B, 1, H_out, W_out, embed_dim)

    # Flatten tokens: (B, tokens, embed_dim)
    # note: reduce_prod on shape[1:-1] -> (1, H_out, W_out) -> product = H_out*W_out
    token_count = tf.reduce_prod(tf.shape(refined)[1:-1])
    patch_embed_flat = tf.reshape(refined, (-1, token_count, embed_dim))

    # CLS token (trainable)
    cls_token = tf.Variable(tf.zeros((1, 1, embed_dim)), name='cls_token', trainable=True)
    patch_embed_flat = tf.concat([
        tf.broadcast_to(cls_token, [tf.shape(patch_embed_flat)[0], 1, embed_dim]),
        patch_embed_flat
    ], axis=1)

    # Positional embedding
    trans = PositionalEmbedding(embed_dim, name='pos_embed')(patch_embed_flat)

    # Transformer layers
    attn_list = []
    for i in range(num_transformer_layers):
        trans, attn = SST(num_heads=num_heads, transformer_layer_depth=2,
                          d_model=embed_dim, dff=dff, dropout_rate=dropout_rate,
                          name=f'transformer_layer_{i}')(trans, training=True)
        # ensure attn is a tensor
        if hasattr(attn, "_values"):
            attn = attn._values
        if isinstance(attn, (list, tuple)):
            attn = attn[0]
        attn = tf.convert_to_tensor(attn)
        attn_list.append(attn)

    # Classification head (use CLS token)
    cls_out = trans[:, 0, :]
    out = layers.Dense(Num_Classes, activation='softmax', name='cls')(cls_out)

    model = Model(inputs, out, name='CSMT')

    # Store attention tensors as non-tracked attributes for visualization
    attn_list = [tf.convert_to_tensor(a) for a in attn_list]
    setattr(model, "_attn_tensors", attn_list)
    setattr(model, "_attn_tensor", attn_list[-1] if len(attn_list) > 0 else None)

    return model


