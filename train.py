import code
import tensorflow as tf


tf.keras.backend.clear_session()
model = code.get_model(code.ConfigUtils().size_augment)
model.summary()
data_train = code.DataGen(True)
data_test = code.DataGen(False)
model.compile(loss=code.dice_coef_loss,optimizer=tf.keras.optimizers.Adam(learning_rate=code.ConfigUtils().lr),metrics=[code.dice_coef, code.iou])
model.fit(data_train, validation_data=data_test, epochs=code.ConfigUtils().number_epoch) # , validation_data=data_test_gener !!!
model.save(code.ConfigPath().path_to_model)