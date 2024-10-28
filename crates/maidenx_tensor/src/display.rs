use std::fmt;

pub fn display_1d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    write!(f, "Tensor(shape=[{}], data=", shape[0])?;
    write!(f, "[")?;
    for (i, val) in data.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?
        }
        write!(f, "{:.4}", val)?;
    }
    write!(f, "])")
}

pub fn display_2d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    writeln!(f, "Tensor(shape=[{}, {}], data=", shape[0], shape[1])?;
    for i in 0..shape[0] {
        if i == 0 {
            write!(f, "[")?;
        }
        write!(f, "[")?;
        for j in 0..shape[1] {
            if j > 0 {
                write!(f, ", ")?
            }
            write!(f, "{:.4}", data[i * shape[1] + j])?;
        }
        if i == shape[0] - 1 {
            writeln!(f, "]])")?;
        } else {
            writeln!(f, "]")?;
        }
    }
    Ok(())
}

pub fn display_3d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    writeln!(
        f,
        "Tensor(shape=[{}, {}, {}], data=",
        shape[0], shape[1], shape[2]
    )?;
    let plane_size = shape[1] * shape[2];

    write!(f, "[")?;
    for i in 0..shape[0] {
        if i > 0 {
            writeln!(f)?;
        }
        write!(f, "[")?;
        for j in 0..shape[1] {
            if j > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for k in 0..shape[2] {
                if k > 0 {
                    write!(f, ", ")?
                }
                write!(f, "{:.4}", data[i * plane_size + j * shape[2] + k])?;
            }
            write!(f, "]")?;
        }
        if i == shape[0] - 1 {
            write!(f, "]])")?;
        } else {
            write!(f, "]")?;
        }
    }
    Ok(())
}

pub fn display_4d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    writeln!(
        f,
        "Tensor(shape=[{}, {}, {}, {}], data=",
        shape[0], shape[1], shape[2], shape[3]
    )?;
    let volume_size = shape[1] * shape[2] * shape[3];
    let plane_size = shape[2] * shape[3];

    write!(f, "[")?;
    for w in 0..shape[0] {
        if w > 0 {
            writeln!(f, ",")?;
        }
        write!(f, "[")?;
        for x in 0..shape[1] {
            if x > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for y in 0..shape[2] {
                if y > 0 {
                    writeln!(f)?;
                }
                write!(f, "[")?;
                for z in 0..shape[3] {
                    if z > 0 {
                        write!(f, ", ")?
                    }
                    write!(
                        f,
                        "{:.4}",
                        data[w * volume_size + x * plane_size + y * shape[3] + z]
                    )?;
                }
                write!(f, "]")?;
            }
            write!(f, "]")?;
        }
        if w == shape[0] - 1 {
            write!(f, "]])")?;
        } else {
            write!(f, "]")?;
        }
    }
    Ok(())
}

pub fn display_nd(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    write!(f, "Tensor(shape=[")?;
    for (i, &dim) in shape.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?
        }
        write!(f, "{}", dim)?;
    }
    write!(f, "], data=[")?;
    for (i, val) in data.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?
        }
        if i >= 8 {
            write!(f, "...")?;
            break;
        }
        write!(f, "{:.4}", val)?;
    }
    write!(f, "])")
}
