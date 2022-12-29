B = 128
E = 150
D = 120
M = 512
H = 8
F = 2048

LAYER_NUM = 6

def multi_head_attention(q, k, v):
    q_fc = M * M
    q_fc_bias = M
    k_fc = M * M
    k_fc_bias = M
    v_fc = M * M
    v_fc_bias = M
    projected_q_fc = B * q * M
    projected_k_fc = B * k * M
    projected_v_fc = B * v * M
    attention_score = B * H * q * k
    attention_weight = B * H * q * k
    attention_output = B * q * M
    fc = M * M
    fc_bias = M
    output = B * q * M

    weight = q_fc + k_fc + v_fc + fc
    bias = q_fc_bias + k_fc_bias + v_fc_bias + fc_bias
    tensor = projected_q_fc + projected_k_fc + projected_v_fc + attention_score + attention_weight + attention_output

    bias = 0

    return (weight + bias + tensor + output) * 4

def add(T):
    return (B * T * M) * 4

def ln(T):
    gamma = M
    beta = M
    deviation = B * T * M
    variance = B * T
    inv_std_dev = B * T
    temp_origin = B * T * M
    temp_normalized = B * T

    output = B * T * M
    # output = 0

    weight = gamma + beta
    tensor = deviation + variance + inv_std_dev
    temp = temp_origin + temp_normalized

    return (weight + tensor + output) * 4

def feedforward(T):
    def fc1(T):
        fc1 = M * F
        fc1_bias = F
        fc1_out = B * T * F
        return (fc1 + fc1_bias + fc1_out) * 4

    def fc2(T):
        fc2 = F * M
        fc2_bias = M
        fc2_out = B * T * M
        return (fc2 + fc2_bias + fc2_out) * 4

    return fc1(T) + fc2(T)

def encoder():
    ln_size = 2 * ln(E)
    return multi_head_attention(E, E, E) + add(E) + feedforward(E) + add(E) + ln_size

def decoder():
    ln_size = 3 * ln(D)
    return multi_head_attention(D, D, D) + add(D) + multi_head_attention(D, E, E) + add(D) + feedforward(D) + add(D) + ln_size


encoder_input = (B * E * M) * 4
decoder_input = (B * D * M) * 4
encoder_size = encoder()
decoder_size = decoder()
ln_size = ln(E) + ln(D)

print(encoder_input + decoder_input + (encoder_size + decoder_size + ln_size) * LAYER_NUM)

# print(multi_head_attention(E, E, E))
# print(add(E))
# print(ln(E))
# print(feedforward(E))