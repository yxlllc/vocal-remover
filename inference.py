import argparse
import os
import yaml
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from lib.nets import CascadedNet
from slicer import Slicer


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def load_sep_model(model_path, device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    model = CascadedNet(
                args.n_fft, 
                args.hop_length, 
                args.n_out, 
                args.n_out_lstm, 
                True, 
                is_mono=args.is_mono)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, args


def split(audio, sample_rate, db_thresh = -40, min_len = 5000):
    slicer = Slicer(
                sr=sample_rate,
                threshold=db_thresh,
                min_length=min_len)    
    chunks = dict(slicer.slice(np.mean(audio, axis=0)))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            result.append(audio[:, int(tag[0]):int(tag[1])])
    return result

    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, required=True)
    p.add_argument('--input', '-i', type=str, required=True)
    p.add_argument('--output_dir', '-o', type=str, default="")
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(args.gpu))
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
    model, model_args = load_sep_model(args.pretrained_model, device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, sr=model_args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast'
    )
    basename = os.path.splitext(os.path.basename(args.input))[0]
    print('done')
    
    if X.ndim == 1:
        if model_args.is_mono:
            X = np.asarray([X])
        else:
            X = np.asarray([X, X])
    
    result_h = np.zeros(0)
    result_hb = np.zeros(0)
    result_he = np.zeros(0)
    result_n = np.zeros(0)
    current_length = 0
    segments = split(X, model_args.sr)
    print('Cut the input audio into ' + str(len(segments)) + ' slices')
    with torch.no_grad():
        for segment in tqdm(segments):
            seg_input = torch.from_numpy(segment).float().unsqueeze(0).to(device)
            seg_output_h, seg_output_hb, seg_output_he = model.predict_fromaudio(seg_input, hb_th=0.001, he_th=0.001)
            seg_output_h = seg_output_h.cpu().numpy()
            seg_output_hb = seg_output_hb.cpu().numpy()
            seg_output_he = seg_output_he.cpu().numpy()
            result_h = np.append(result_h, seg_output_h)
            result_hb = np.append(result_hb, seg_output_hb)
            result_he = np.append(result_he, seg_output_he)
            print('validating output directory...', end=' ')
    result_n = X - result_h
    
    output_dir = args.output_dir
    if output_dir != "":  # modifies output_dir if theres an arg specified
        output_dir = output_dir.rstrip('/') + '/'
        os.makedirs(output_dir, exist_ok=True)
    print('done')
    
    sf.write('{}{}_harmonic.wav'.format(output_dir, basename), result_h.T, sr)
    sf.write('{}{}_harmonic_base.wav'.format(output_dir, basename), result_hb.T, sr)
    sf.write('{}{}_harmonic_even.wav'.format(output_dir, basename), result_he.T, sr)
    sf.write('{}{}_noise.wav'.format(output_dir, basename), result_n.T, sr)