

## Downloading datasets

### ASCADv1-fixed (cropped and uncropped)

Based on the instructions [here](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key): download both the cropped and uncropped versions of the dataset by navigating to the desired directory and running ```wget https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e -O ASCAD_data.zip```Verify the checksum with the line```sha256 a6884faf97133f9397aeb1af247dc71ab7616f3c181190f127ea4c474a0ad72c ASCAD_data.zip```Extract to desired directory and verify that the following files exist: `ASCAD_databases/ASCAD.h5`, `ASCAD_databases/ASCAD_desync50.h5`, `ASCAD_databases/ASCAD_desync100.h5`, `ASCAD_databases/ATMega8515_raw_traces.h5`.