use crate::error::CpuResult;

pub fn linear_forward(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch_size: i32,
    out_features: i32,
    in_features: i32,
) -> CpuResult<Vec<f32>> {
    let batch_size = batch_size as usize;
    let out_features = out_features as usize;
    let in_features = in_features as usize;

    let mut output = vec![0.0; batch_size * out_features];

    for b in 0..batch_size {
        for o in 0..out_features {
            let mut sum = 0.0;
            for i in 0..in_features {
                sum += input[b * in_features + i] * weight[o * in_features + i];
            }
            if let Some(bias) = bias {
                sum += bias[o];
            }
            output[b * out_features + o] = sum;
        }
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn bilinear_forward(
    input1: &[f32],
    input2: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch_size: i32,
    out_features: i32,
    in1_features: i32,
    in2_features: i32,
) -> CpuResult<Vec<f32>> {
    let batch_size = batch_size as usize;
    let out_features = out_features as usize;
    let in1_features = in1_features as usize;
    let in2_features = in2_features as usize;

    let mut output = vec![0.0; batch_size * out_features];

    for b in 0..batch_size {
        for o in 0..out_features {
            let mut sum = 0.0;
            for i in 0..in1_features {
                for j in 0..in2_features {
                    sum += input1[b * in1_features + i]
                        * weight[o * in1_features * in2_features + i * in2_features + j]
                        * input2[b * in2_features + j];
                }
            }
            if let Some(bias) = bias {
                sum += bias[o];
            }
            output[b * out_features + o] = sum;
        }
    }

    Ok(output)
}
