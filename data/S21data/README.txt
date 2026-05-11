The raw S21 data is stored in Ivan's group drive under G:\AS-Filer\PHY\ivpechen\General\Users\JT\JT\CoreScansFilmStressPaper. 
The file layout proceeds from sample name -> measurement date -> measurementlabel -> field -> temperature
Example: G:\AS-Filer\PHY\ivpechen\General\Users\JT\JT\CoreScansFilmStressPaper\SU_Nb_10mTorr\250501\pwrDependenceCorrectAtt\-0uT\100mK

Each S21 scan produces 3 files:
1. A .DAT file containing the raw scan data in 5 columns: frequency, magnitude, phase, real, imaginary
2. A .JSON file containing all the scan parameters: IF (Hz), avg, centerFreq (GHz), faaTemp (K), numPoints, power (dBm), span (MHz). fieldStrength and measuredCurr are unused
3. a .PNG file displaying the measurement plotted in Python

After the scan is taken, the data is passed to our fitting algorithm based on Megrant 2012, circle-fitting the inverse S21 and deriving Q parameters.
The fitting results are then saved as two files into a folder labeled as ResX_P=-Y, where X is the resonator number and Y is the input power:
1. A .DAT file of the fitparams with the following entries: f0, theta, Qc, Qi, Q, f0_SD, theta_SD, Qc_SD, Qi_SD, Q_SD
2. A .PNG of the scan results showing a geometric fit (blue) and and the more refined Megrant fit (purple) on multiple different formats of the scan
