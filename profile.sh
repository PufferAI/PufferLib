nsys profile --force-overwrite true --capture-range=cudaProfilerApi --cuda-graph-trace=node --sample=none -o profile python -m pufferlib.pufferl train breakout --vec.num-buffers 1 --profile True
echo "All kernels"
nsys stats --report cuda_gpu_kern_sum:base --force-export=true profile.nsys-rep
echo "NVTX tags"
nsys stats --report nvtx_sum --force-export=true profile.nsys-rep
echo "Kernel groups"
nsys export --type=sqlite --force-overwrite=true -o profile.sqlite profile.nsys-rep
sqlite3 -header -column profile.sqlite "
  SELECT
    CASE
      WHEN s.value LIKE '%gemm%' OR s.value LIKE '%Kernel2%' OR s.value LIKE '%splitKreduce%' THEN 'matmul'
      WHEN s.value LIKE '%ppo_loss%' OR s.value LIKE '%ppo_var_mean%' THEN 'ppo_loss'
      WHEN s.value LIKE '%mingru%' THEN 'mingru'
      WHEN s.value LIKE '%muon%' THEN 'muon'
      WHEN s.value LIKE '%cast%' OR s.value LIKE '%select_copy%' OR s.value LIKE '%index_copy%' OR s.value LIKE '%transpose%' THEN 'copy'
      WHEN s.value LIKE '%sample_logits%' OR s.value LIKE '%multinomial%' THEN 'sample'
      WHEN s.value LIKE '%puff_advantage%' THEN 'advantage'
      WHEN s.value LIKE '%prio%' THEN 'prio_replay'
      WHEN s.value LIKE '%assemble_decoder%' THEN 'decoder_grad'
      ELSE s.value
    END AS group_name,
    COUNT(*) AS count,
    ROUND(100.0 * SUM(k.end - k.start) / (SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL), 1) AS 'time_%',
    ROUND(SUM(k.end - k.start) / 1e6, 2) AS total_ms,
    ROUND(AVG(k.end - k.start) / 1e3, 2) AS avg_us
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON k.shortName = s.id
  GROUP BY group_name
  ORDER BY total_ms DESC;
"
