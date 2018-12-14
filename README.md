# Exemplar voice conversion

Exemplar-based voice conversion  
### 1. dict  
### 2. NN attempt for fw  
### 3. convert  
### 4. calculate some objective result

## Update 2018 Dec 14:
#### Big update: The old version seems to be wrong at first attempt: 
1. The DTW is not used for time-series data. It must be used on spectra-features, depends on whether it is DFW, AFW or CFW
2. W is not a function, it is a MATRIX belong to R^(MxN)

#### Unresolved problem:
1. Moving average filter

#### What changes:
1. Old code will  be moved to *old_code* directory
2. New code will take over, with the same name


