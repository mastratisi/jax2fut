let my_f (Var(id=4750816960):float32[5]: f32[5]) =
  let Var(id=4749379072):float32[5] = sin(Var(id=4750816960):float32[5])
  let Var(id=4749308736):float32[5] = (Var(id=4749379072):float32[5] + Var(id=4750816960):float32[5])
  in Var(id=4749308736):float32[5]