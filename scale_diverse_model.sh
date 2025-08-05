# 评测不同模型架构的泛化性能
# 轻量级模型
python FL_train.py --alg fedrot-local --model simple-cnn --dataset RS-5 --noise_rate 0.8 --noise_pattern symmetric
python FL_train.py --alg fedrot-local --model resnet18 --dataset RS-5 --noise_rate 0.8 --noise_pattern symmetric
python FL_train.py --alg fedrot-local --model resnet34 --dataset RS-5 --noise_rate 0.8 --noise_pattern symmetric

# 中等复杂度模型
python FL_train.py --alg fedrot-local --model resnet50 --dataset RS-5 --noise_rate 0.8 --noise_pattern symmetric
python FL_train.py --alg fedrot-local --model vgg16 --dataset RS-5 --noise_rate 0.8 --noise_pattern symmetric
python FL_train.py --alg fedrot-local --model densenet121 --dataset RS-5 --noise_rate 0.8 --noise_pattern symmetric

# 预训练模型
python FL_train.py --alg fedrot-local --model resnet18-pretrained --dataset RS-5 --noise_rate 0.8 --noise_pattern symmetric
python FL_train.py --alg fedrot-local --model resnet50-pretrained --dataset RS-5 --noise_rate 0.8 --noise_pattern symmetric
python FL_train.py --alg fedrot-local --model vit-b-16 --dataset RS-5 --noise_rate 0.8 --noise_pattern symmetric