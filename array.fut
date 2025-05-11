let array_ops_example (x0: [4]f32) =
  let x1: [4]f32 = map (\fvar: f32 -> let lvar: f32 = (f32.**) 2.0 fvar in lvar)  x0 in
  let x2: f32 = f32.sum(x1) in
  let x3: f32 = f32.sqrt(x2) in
 x3
