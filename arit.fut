let arithmetic_example (x0: f32, x1: f32) =
  let x2: f32 = f32.sin(x0) in
  let x3: f32 = f32.cos(x1) in
  let x4: f32 = (f32.*) (x3) (2.0) in
  let x5: f32 = (f32.+) (x2) (x4) in
 x5

