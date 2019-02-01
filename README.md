# Exemplar voice conversion

### Todo list

- [ ] 01_make_dict_parallel.py
    + [x] Done in utils.py. Add parameter `parallel` to `io_read_speaker_data`
    + [x] _extract_features: Extract mcep
    + [ ] _dtw_alignment (this consume most of the time)

- [ ] Documentation for important function
- [ ] ...

Exemplar-based voice conversion  
### 1. dict  
### 2. NN attempt for fw  
### 3. convert  
### 4. calculate some objective result

## Update 2019 Jan 17:
1. Switch to simpler version of exemplar-based method: implement non-compensation version first. Will release 1.0 before Feb
    Source: - http://www.zhizheng.org/slides/SSW2013_poster_nmf.pdf
            - Zhizheng Wu et al., Exemplar-based voice conversion using non-negative spectrogram deconvolution

## Update 2019 Jan 15:
1. lambda declaration is replaced (<del>lambda x, y: np.linalg.norm(x - y, ord=1)</del>) for distance function

## Update 2018 Dec 14:
#### Big update: The old version seems to be wrong at first attempt: 
1. <del>The DTW is not used for time-series data. It must be used on spectra-features, depends on whether it is DFW, AFW or CFW</del> DTW is first applied on time series data to get a parallel training set, then either DFW, AFW or CFW is applied to that parallel data

2. W is not a function, it is a MATRIX belong to R^(MxN)

#### Unresolved problem:
1. Moving average filter

#### What changes:
1. Old code will  be moved to *old_code* directory
2. New code will take over, with the same name
