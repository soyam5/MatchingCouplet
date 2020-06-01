项目名称： 对对联项目
主要功能：
	1.实现通过上联，得出下联的功能
	2.实现了通过图片识别出文字和通过图片意境匹配出对联的功能
实现原理：
	rnn模型：使用tf.keras.Sequential生成两层的模型
	seq2seq-attention模型：实现encoder、decoder两个模型，encoder生成对于输入的sentence的encoding，decoder在encoding基础上生成目标sentence
	基于百度api的调用图片识别和意境识别出文字
