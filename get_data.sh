export KAGGLE_USERNAME=hngcao
export KAGGLE_KEY=341299e2aec23fac5fc6907990da54aa

if [ $# -ne 1 ]; then
    echo "Usage: $0 <data_path>"
    exit 1
fi

# Get the data_path argument
data_path="$1"

# Check if the directory already exists
if [ -d "$data_path" ]; then
    echo "Directory '$data_path' already exists."
    exit 1
fi

mkdir "$data_path"
echo "Directory '$data_path' created successfully."


cd $data_path
kaggle competitions download -c inria-bci-challenge
unzip inria-bci-challenge.zip
rm inria-bci-challenge.zip

mkdir preproc
mkdir train
mkdir test

unzip train.zip -d train
unzip test.zip -d test
rm *.zip
cp TrainLabels.csv train

