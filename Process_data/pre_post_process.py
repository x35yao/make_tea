from pre_process import Pre_process
from post_process import Post_process
raw_dir = './raw_data/2022-10-27'
# prep = Pre_process(raw_dir)
# prep.process_servo()
# prep.process_ndi()
# prep.process_video()
pre_dir = raw_dir.replace('raw_data', 'preprocessed')
postp = Post_process(pre_dir)
postp.process_ndi()
postp.process_video()
postp.process_servo()
postp.save_processed_file()