# LiDAR–IMU pose learning

Entrenamiento autosupervisado de movimiento planar para el vehículo APEX. La
red combina una rama local con los dos últimos barridos y una rama causal de
contexto con cinco barridos. Ambas comparten la CNN espacial, pero tienen GRU
independientes. Una compuerta aprendida decide cuánto aplicar de la corrección
contextual. Las muestras IMU cubren los intervalos correspondientes. Produce:

- `delta_pose = [dx, dy, dyaw]`;
- incertidumbre diagonal de ese incremento;
- sesgos estimados `[bax, bay, baz, bgx, bgy, bgz]`.

La traslación incorpora una mecanización recurrente diferenciable:

```text
v[t]  = v[t-1] + (a[t] - bias_a[t]) dt + delta_v_net[t]
dp[t] = v[t-1] dt + 0.5 (a[t] - bias_a[t]) dt^2 + delta_p_net[t]
```

La GRU inicializa la velocidad de cada ventana y aprende únicamente las
correcciones residuales. La rotación usa de forma análoga la integral del
giróscopo corregida por el sesgo estimado.

Las lecturas se sincronizan por sus timestamps reales: cada intervalo entre
dos barridos conserva las muestras IMU que realmente llegaron, sin forzarlas
a un número interpolado. Para formar lotes se rellena hasta el máximo del
intervalo y se entrega la longitud válida a una GRU empaquetada, que ignora el
relleno. El tensor LiDAR conserva los 360 rayos de cada uno de los cinco
barridos de contexto.

La corrección LiDAR se forma como:

```text
delta_pose = delta_pair + sigmoid(gate) * delta_context + delta_inertial
```

Además de las pérdidas por paso, la composición de los seis incrementos de una
secuencia debe coincidir con la composición de los incrementos obtenidos solo
desde sensores. Esto impide que las dos ramas produzcan movimientos locales
incompatibles con el contexto largo.

La pose exacta de Gazebo **no entra en el entrenamiento ni en la selección del
checkpoint**. Durante entrenamiento se utiliza movimiento relativo calculado
solo desde LiDAR mediante ICP implementado en PyTorch (`torch.cdist` y
`torch.linalg.svd`), consistencia con la integral giroscópica,
suavidad temporal y regularización de los sesgos. El ground truth se abre una
única vez al terminar para evaluar y dibujar el 15 % de validación. El 15 % de
test permanece reservado y solo se informa su pérdida autosupervisada.

El modo opcional `--supervised-ground-truth` sustituye esos objetivos ICP por
incrementos relativos obtenidos de la pose exacta. Está pensado para medir el
techo supervisado y estudiar generalización entre pistas y sensores.

El reparto es cronológico para evitar que ventanas solapadas de una misma
vuelta aparezcan a ambos lados de una partición:

- 70 % entrenamiento;
- 15 % validación;
- 15 % test.

Ejemplo:

```bash
python3 learning/lidar_imu_pose/train_pose_fusion.py \
  --run-dir simulation/data/fusion_research/sensor_lap_final_20260924_173000 \
  --epochs 30 --device cuda
```

Para entrenar directamente contra la pose exacta del simulador y generar las
curvas de aprendizaje:

```bash
python3 learning/lidar_imu_pose/train_pose_fusion.py \
  --run-dir simulation/data/fusion_research/sensor_lap_final_20260924_173000 \
  --output-dir simulation/learning/outputs/lidar_imu_pose_supervised \
  --epochs 40 --device cuda --supervised-ground-truth
```

Los checkpoints, cachés, métricas y figuras se escriben por defecto en
`learning/outputs/lidar_imu_pose/`, que está ignorado por Git.
