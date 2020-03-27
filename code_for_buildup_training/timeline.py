from tensorflow.python.client import timeline

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()


sess.run([train_OP], options=options, run_metadata=run_metadata)

# 保存为json文件
fetched_timeline = timeline.Timeline(run_metadata.step_stats)
chrome_trace = fetched_timeline.generate_chrome_trace_format()
with open('timeline.json', 'w') as f:
    f.write(chrome_trace)