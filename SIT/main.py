import os
import argparse
import tqdm
from utils import *
from attack import *

def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=64, type=int, help='the bacth size(train)')
    parser.add_argument('--eval_batchsize', default=8, type=int, help='the bacth size(eval)')
    
    parser.add_argument('--eps', default=16/255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6/255, type=float, help='the stepsize to update the perturbation')
    
    parser.add_argument('--num_copies', default=20, type=int, help='number of copies')
    parser.add_argument('--num_block', default=3, type=int, help='number of shuffled blocks')
    
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model', choices=['resnet18', 'resnet101', 'resnext50','densenet121', 'mobilenet', 'vit', 'swin','inceptionv3'])
    
    parser.add_argument('--input_dir', default='data/', type=str, help='the path for the benign images')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--output_txt',default='trans.txt',type=str,help='the path to store the results')
    return parser.parse_args()


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s

def main():
    args = get_parser()
    print(print_args(args, []))
    f2l = load_labels(os.path.join(os.getcwd(),args.input_dir, 'val_rs.csv')) #라벨 로드
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    if not args.eval: # attack모드
        model = wrap_model(model_list[args.model](weights='DEFAULT').eval().cuda())
        attacker = SIA(model, args.eps, args.alpha, args.epoch, args.momentum,args.num_copies,args.num_block)
        
        # 배치 단위로 이미지 로드 및 공격 수행
        for batch_idx, [filenames, images] in tqdm.tqdm(enumerate(load_images(os.path.join(os.getcwd(),args.input_dir,'data'), args.batchsize))):# 배치 단위로 이미지 로드 및 공격 수행
            #attacker.gsnr=None
            labels = get_labels(filenames, f2l)
            perturbations = attacker(images, labels)
            save_images(args.output_dir, images + perturbations.cpu(), filenames)
        
        # adversarial images에 대해 얼마나 정확하게 예측했는지
        accuracy = dict() 
        res = 'Model | Accuracy\n'
        res += '-' * 30 + '\n'
        for model_name, model_arc in model_list.items():
            # model = wrap_model(model_list[args.model](weights='DEFAULT').eval().cuda())
            model = wrap_model(model_arc(weights='DEFAULT').eval().cuda())
            succ, total = 0, 0
            for batch_idx, (filenames, images) in tqdm.tqdm(enumerate(load_images(args.output_dir, args.eval_batchsize))):
                labels = get_labels(filenames, f2l)
                pred = model(images.cuda())
                succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum() #올바르게 예측한 이미지
                total += labels.shape[0]
            accuracy[model_name] = (succ / total * 100)
            accuracy[model_name]=100-accuracy[model_name]
            print(model_name, accuracy[model_name])
            res += '{} | {:.2f}%\n'.format(model_name, accuracy[model_name])
        
        #결과 저장 및 출력
        print(accuracy) 
        print(res)
        with open(args.output_txt,'a+') as f:
            f.write(res)
    
    else:
        accuracy = {}
        res = 'Model | Accuracy\n'
        res += '-' * 30 + '\n'
        for model_name, model_arc in model_list.items():
            if isinstance(model_arc, int):
                continue
            model = wrap_model(model_arc(weights='DEFAULT').eval().cuda())  # 모델 로드 및 랩핑
            succ, total = 0, 0
            for batch_idx, (filenames, images) in tqdm.tqdm(enumerate(load_images(args.output_dir, args.batchsize))):
                labels = get_labels(filenames, f2l)  # 이미지에 대한 라벨 가져오기
                pred = model(images.cuda())  # 모델 예측 수행
                succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()  # 성공 횟수 계산
                total += labels.shape[0]
            accuracy[model_name] = (succ / total * 100)
            print(model_name, accuracy[model_name])
            res += '{} | {:.2f}%\n'.format(model_name, accuracy[model_name])
        
        # 결과 저장 및 출력
        print(accuracy)
        print(res)
        with open(args.output_txt, 'a+') as f:
            f.write(res)
            f.write('\r\n')

            
if __name__ == '__main__':
    main()
    