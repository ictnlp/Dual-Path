src = SOURCE_LANGUAGE
tgt = TARGET_LANGUAGE
train_data = DIR_TO_TRAIN_DATA
vaild_data = DIR_TO_VALID_DATA
test_data = DIR_TO_TEST_DATA
data=PATH_TO_DATA

fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
    --trainpref ${train_data} --validpref ${vaild_data} \
    --testpref ${test_data}\
    --destdir ${data} \
    --workers 20