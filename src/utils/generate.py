'''
This script assumes c-version STRAIGHT which is not available to public. Please use your
own vocoder to replace this script.
'''
import sys, os, subprocess, glob, subprocess
#from utils import GlobalCfg

from io_funcs.binary_io import  BinaryIOCollection
import numpy as np
import multiprocessing as mp
import logging
import pdb
import pyworld
import pysptk
import librosa
import scipy
#import configuration

# cannot have these outside a function - if you do that, they get executed as soon
# as this file is imported, but that can happen before the configuration is set up properly
# SPTK     = cfg.SPTK
# NND      = cfg.NND
# STRAIGHT = cfg.STRAIGHT
def read_binfile(filename, dim=60, dtype=np.float64):
    '''
    Reads binary file into numpy array.
    '''
 #   pdb.set_trace()
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=dtype)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)).astype('float64') # This is to keep compatibility with numpy default dtype.
    m_data = np.squeeze(m_data)
    return  m_data

def run_process(args,log=True):
    # pdb.set_trace()
    logger = logging.getLogger("subprocess")

    # a convenience function instead of calling subprocess directly
    # this is so that we can do some logging and catch exceptions

    # we don't always want debug logging, even when logging level is DEBUG
    # especially if calling a lot of external functions
    # so we can disable it by force, where necessary
    if log:
        logger.debug('%s' % args)

    try:
        # the following is only available in later versions of Python
        # rval = subprocess.check_output(args)

        # bufsize=-1 enables buffering and may improve performance compared to the unbuffered case
        p = subprocess.Popen(args, bufsize=-1, shell=True,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        close_fds=True, env=os.environ)
        # better to use communicate() than read() and write() - this avoids deadlocks
        (stdoutdata, stderrdata) = p.communicate()

        if p.returncode != 0:
            # for critical things, we always log, even if log==False
            logger.critical('exit status %d' % p.returncode )
            logger.critical(' for command: %s' % args )
            logger.critical('      stderr: %s' % stderrdata )
            logger.critical('      stdout: %s' % stdoutdata )
            raise OSError

        return (stdoutdata, stderrdata)

    except subprocess.CalledProcessError as e:
        # not sure under what circumstances this exception would be raised in Python 2.6
        logger.critical('exit status %d' % e.returncode )
        logger.critical(' for command: %s' % args )
        # not sure if there is an 'output' attribute under 2.6 ? still need to test this...
        logger.critical('  output: %s' % e.output )
        raise

    except ValueError:
        logger.critical('ValueError for %s' % args )
        raise

    except OSError:
        logger.critical('OSError for %s' % args )
        raise

    except KeyboardInterrupt:
        logger.critical('KeyboardInterrupt during %s' % args )
        try:
            # try to kill the subprocess, if it exists
            p.kill()
        except UnboundLocalError:
            # this means that p was undefined at the moment of the keyboard interrupt
            # (and we do nothing)
            pass

        raise KeyboardInterrupt


def bark_alpha(sr):
    return 0.8517*np.sqrt(np.arctan(0.06583*sr/1000.0))-0.1916

def erb_alpha(sr):
    return 0.5941*np.sqrt(np.arctan(0.1418*sr/1000.0))+0.03237

def post_filter(mgc_file_in, mgc_file_out, mgc_dim, pf_coef, fw_coef, co_coef, fl_coef, gen_dir, cfg):

    SPTK = cfg.SPTK

    line = "echo 1 1 "
    for i in range(2, mgc_dim):
        line = line + str(pf_coef) + " "
    # pdb.set_trace()
    run_process('{line} | {x2x} +af > {weight}'.format(line=line, x2x=SPTK['X2X'], weight=os.path.join(gen_dir, 'weight')))

    run_process('{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | {c2acr} -m {co} -M 0 -l {fl} > {base_r0}'.format(freqt=SPTK['FREQT'], order=mgc_dim-1, fw=fw_coef, co=co_coef, mgc=mgc_file_in, c2acr=SPTK['C2ACR'], fl=fl_coef, base_r0=mgc_file_in+'_r0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {freqt} -m {order} -a {fw} -M {co} -A 0 | {c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        freqt=SPTK['FREQT'], fw=fw_coef, co=co_coef,
                        c2acr=SPTK['C2ACR'], fl=fl_coef, base_p_r0=mgc_file_in+'_p_r0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 0 -e 0 > {base_b0}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        mc2b=SPTK['MC2B'], fw=fw_coef,
                        bcp=SPTK['BCP'], base_b0=mgc_file_in+'_b0'))

    run_process('{vopr} -d < {base_r0} {base_p_r0} | {sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
                .format(vopr=SPTK['VOPR'], base_r0=mgc_file_in+'_r0', base_p_r0=mgc_file_in+'_p_r0',
                        sopr=SPTK['SOPR'],
                        base_b0=mgc_file_in+'_b0', base_p_b0=mgc_file_in+'_p_b0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 1 -e {order} | {merge} -n {order2} -s 0 -N 0 {base_p_b0} | {b2mc} -m {order} -a {fw} > {base_p_mgc}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        mc2b=SPTK['MC2B'],  fw=fw_coef,
                        bcp=SPTK['BCP'],
                        merge=SPTK['MERGE'], order2=mgc_dim-2, base_p_b0=mgc_file_in+'_p_b0',
                        b2mc=SPTK['B2MC'], base_p_mgc=mgc_file_out))

    return

def wavgen_straight_type_vocoder(gen_dir, file_id_list, cfg, logger):
    '''
    Waveform generation with STRAIGHT or WORLD vocoders.
    (whose acoustic parameters are: mgc, bap, and lf0)
    '''
    SPTK     = cfg.SPTK
#    NND      = cfg.NND
    STRAIGHT = cfg.STRAIGHT
    WORLD    = cfg.WORLD

    ## to be moved
    pf_coef = cfg.pf_coef
    if isinstance(cfg.fw_alpha, str):
        if cfg.fw_alpha=='Bark':
            fw_coef = bark_alpha(cfg.sr)
        elif cfg.fw_alpha=='ERB':
            fw_coef = bark_alpha(cfg.sr)
        else:
            raise ValueError('cfg.fw_alpha='+cfg.fw_alpha+' not implemented, the frequency warping coefficient "fw_coef" cannot be deduced.')
    else:
        fw_coef = cfg.fw_alpha
    co_coef = cfg.co_coef
    fl_coef = cfg.fl

    if cfg.apply_GV:
        io_funcs = BinaryIOCollection()

        logger.info('loading global variance stats from %s' % (cfg.GV_dir))

        ref_gv_mean_file = os.path.join(cfg.GV_dir, 'ref_gv.mean')
        gen_gv_mean_file = os.path.join(cfg.GV_dir, 'gen_gv.mean')
        ref_gv_std_file  = os.path.join(cfg.GV_dir, 'ref_gv.std')
        gen_gv_std_file  = os.path.join(cfg.GV_dir, 'gen_gv.std')

        ref_gv_mean, frame_number = io_funcs.load_binary_file_frame(ref_gv_mean_file, 1)
        gen_gv_mean, frame_number = io_funcs.load_binary_file_frame(gen_gv_mean_file, 1)
        ref_gv_std, frame_number = io_funcs.load_binary_file_frame(ref_gv_std_file, 1)
        gen_gv_std, frame_number = io_funcs.load_binary_file_frame(gen_gv_std_file, 1)

    # counter=1
    max_counter = len(file_id_list)

    for filename in file_id_list:
 #       logger.info('creating waveform for %4d of %4d: %s' % (max_counter, filename))
        # counter=counter+1
      #  pdb.set_trace()
        base   = filename
        files = {'sp'  : base + cfg.sp_ext,
                 'mgc' : base + cfg.mgc_ext,
                 'f0'  : base + '.f0',
                 'lf0' : base + cfg.lf0_ext,
                 'ap'  : base + '.ap',
                 'bap' : base + cfg.bap_ext,
                 'wav' : base + '.wav'}

        mgc_file_name = files['mgc']
        bap_file_name = files['bap']
     #   pdb.set_trace()
        cur_dir = os.getcwd()
        os.chdir(gen_dir)

        ### post-filtering
        if cfg.do_post_filtering:
            mgc_file_name = files['mgc']+'_p_mgc'
            # pdb.set_trace()
            post_filter(files['mgc'], mgc_file_name, cfg.mgc_dim, pf_coef, fw_coef, co_coef, fl_coef, gen_dir, cfg)

        if cfg.vocoder_type == "STRAIGHT" and cfg.apply_GV:
            gen_mgc, frame_number = io_funcs.load_binary_file_frame(mgc_file_name, cfg.mgc_dim)

            gen_mu  = np.reshape(np.mean(gen_mgc, axis=0), (-1, 1))
            gen_std = np.reshape(np.std(gen_mgc, axis=0), (-1, 1))

            local_gv = (ref_gv_std/gen_gv_std) * (gen_std - gen_gv_mean) + ref_gv_mean;

            enhanced_mgc = np.repeat(local_gv, frame_number, 1).T / np.repeat(gen_std, frame_number, 1).T * (gen_mgc - np.repeat(gen_mu, frame_number, 1).T) + np.repeat(gen_mu, frame_number, 1).T;

            new_mgc_file_name = files['mgc']+'_p_mgc'
            io_funcs.array_to_binary_file(enhanced_mgc, new_mgc_file_name)

            mgc_file_name = files['mgc']+'_p_mgc'

        if cfg.do_post_filtering and cfg.apply_GV:
            logger.critical('Both smoothing techniques together can\'t be applied!!\n' )
           # raise

        ###mgc to sp to wav
        if cfg.vocoder_type == 'STRAIGHT':
            run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} > {sp}'
                        .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.mgc_dim-1, fl=cfg.fl, mgc=mgc_file_name, sp=files['sp']))
            run_process('{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} > {f0}'.format(sopr=SPTK['SOPR'], lf0=files['lf0'], f0=files['f0']))
            run_process('{x2x} +fa {f0} > {f0a}'.format(x2x=SPTK['X2X'], f0=files['f0'], f0a=files['f0'] + '.a'))

            if cfg.use_cep_ap:
                run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 0 {bap} > {ap}'
                            .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.bap_dim-1, fl=cfg.fl, bap=files['bap'], ap=files['ap']))
            else:
                run_process('{bndap2ap} {bap} > {ap}'
                             .format(bndap2ap=STRAIGHT['BNDAP2AP'], bap=files['bap'], ap=files['ap']))

            run_process('{synfft} -f {sr} -spec -fftl {fl} -shift {shift} -sigp 1.2 -cornf 4000 -float -apfile {ap} {f0a} {sp} {wav}'
                        .format(synfft=STRAIGHT['SYNTHESIS_FFT'], sr=cfg.sr, fl=cfg.fl, shift=cfg.shift, ap=files['ap'], f0a=files['f0']+'.a', sp=files['sp'], wav=files['wav']))

            run_process('rm -f {sp} {f0} {f0a} {ap}'
                        .format(sp=files['sp'],f0=files['f0'],f0a=files['f0']+'.a',ap=files['ap']))
        elif cfg.vocoder_type == 'WORLD':

            run_process('{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | {x2x} +fd > {f0}'.format(sopr=SPTK['SOPR'], lf0=files['lf0'], x2x=SPTK['X2X'], f0=files['f0']))

            run_process('{sopr} -c 0 {bap} | {x2x} +fd > {ap}'.format(sopr=SPTK['SOPR'],bap=files['bap'],x2x=SPTK['X2X'],ap=files['ap']))

            ### If using world v2, please comment above line and uncomment this
            #run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 0 {bap} | {sopr} -d 32768.0 -P | {x2x} +fd > {ap}'
            #            .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.bap_dim, fl=cfg.fl, bap=bap_file_name, sopr=SPTK['SOPR'], x2x=SPTK['X2X'], ap=files['ap']))

            run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} | {sopr} -d 32768.0 -P | {x2x} +fd > {sp}'
                        .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.mgc_dim-1, fl=cfg.fl, mgc=mgc_file_name, sopr=SPTK['SOPR'], x2x=SPTK['X2X'], sp=files['sp']))

            run_process('{synworld} {fl} {sr} {f0} {sp} {ap} {wav}'
                         .format(synworld=WORLD['SYNTHESIS'], fl=cfg.fl, sr=cfg.sr, f0=files['f0'], sp=files['sp'], ap=files['ap'], wav=files['wav']))
            # y = pw.synthesize(f0, sp, ap, fs)

            run_process('rm -f {ap} {sp} {f0}'.format(ap=files['ap'],sp=files['sp'],f0=files['f0']))
        elif cfg.vocoder_type == 'WORLD_PY':
            logging.info("generate speech with py world, sampling rate is {0}".format(cfg.sr))
            # pdb.set_trace()
            lf0_file = files['lf0']
            lf0 = read_binfile(lf0_file,dim=1,dtype=np.float32)
            zeros_index = np.where(lf0 == -1E+10)
            nonzeros_index = np.where(lf0 != -1E+10)
            f0=lf0.copy()
            f0[zeros_index] = 0
            f0[nonzeros_index] = np.exp(lf0[nonzeros_index])
            f0 = f0.astype(np.float64)
            if cfg.sr==16000:
                bap_dim = 1
            elif cfg.sr == 48000:
                bap_dim = 5
            else:
                pass
            bap = read_binfile(bap_file_name,dim=bap_dim,dtype=np.float32)
            ap = pyworld.decode_aperiodicity(bap.astype(np.float64).reshape(-1,bap_dim), cfg.sr, cfg.fl)
            mc = read_binfile(mgc_file_name,dim=60,dtype=np.float32)
            alpha = pysptk.util.mcepalpha(cfg.sr)
            sp = pysptk.mc2sp(mc.astype(np.float64), fftlen=cfg.fl, alpha=alpha)
            wav = pyworld.synthesize(f0,sp,ap, cfg.sr,5)
            x2 = wav * 32768
            x2 = x2.astype(np.int16)
            scipy.io.wavfile.write(files['wav'], cfg.sr, x2)
            os.chdir(cur_dir)
   # pool = mp.Pool(mp.cpu_count())
   # pool.map(_synthesis, file_id_list)


def generate_wav(gen_dir, file_id_list, cfg):

    logger = logging.getLogger("wav_generation")

    ## STRAIGHT or WORLD vocoders:
    if (cfg.vocoder_type=='STRAIGHT') or (cfg.vocoder_type=='WORLD') or (cfg.vocoder_type=='WORLD_PY'):
        wavgen_straight_type_vocoder(gen_dir, file_id_list, cfg, logger)

    # Add your favorite vocoder here.

    # If vocoder is not supported:
    else:
        logger.critical('The vocoder %s is not supported yet!\n' % cfg.vocoder_type )
    return