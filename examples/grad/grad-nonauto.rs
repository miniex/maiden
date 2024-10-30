use maidenx::prelude::*;
use std::f32;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_current_device(Device::cpu())?;

    let x = Tensor::linspace(-f32::consts::PI, f32::consts::PI, 2000)?;
    let y = x.sin()?;

    let mut a = Tensor::randn(&[1])?;
    let mut b = Tensor::randn(&[1])?;
    let mut c = Tensor::randn(&[1])?;
    let mut d = Tensor::randn(&[1])?;

    let start = Instant::now();

    let learning_rate = 1e-6;
    for t in 1..10000 {
        let x0 = a.item()?;
        let x1 = x.scalar_mul(b.item()?)?;
        let x2 = x.pow(2.0)?.scalar_mul(c.item()?)?;
        let x3 = x.pow(3.0)?.scalar_mul(d.item()?)?;

        let y_pred = x3.add(&x2)?.add(&x1)?.scalar_add(x0)?;
        let loss = y_pred.sub(&y)?.pow(2.0)?.sum()?.item()?;

        if t != 0 && t % 100 == 0 {
            println!("t: {}, loss: {}", t, loss);
        }

        let grad_y_pred = (2.0 * &y_pred.sub(&y)?)?;
        let grad_a = grad_y_pred.sum()?;
        let grad_b = (&grad_y_pred * &x)?.sum()?;
        let grad_c = (&grad_y_pred * &x.pow(2.0)?)?.sum()?;
        let grad_d = (&grad_y_pred * &x.pow(3.0)?)?.sum()?;

        a = a.sub(&(learning_rate * &grad_a)?)?;
        b = b.sub(&(learning_rate * &grad_b)?)?;
        c = c.sub(&(learning_rate * &grad_c)?)?;
        d = d.sub(&(learning_rate * &grad_d)?)?;
    }

    let elapsed = start.elapsed();

    println!(
        "Result: y = {} + {}x + {}x^2 + {}x^3",
        a.item()?,
        b.item()?,
        c.item()?,
        d.item()?
    );
    println!("Time: {:?}", elapsed);

    Ok(())
}
