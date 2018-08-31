experiment="synthetic"
for dataset in configs/1.synthetic/*
do
    for file in configs/1.synthetic/$dataset/*
    do
        file_path=configs/1.synthetic/$dataset/$file
        model=${file%".yaml"}
        flow-submit bin/1.synthetic/submit/$dataset/$model.sh orion hunt -n $experiment.$dataset.$model
    done
done
