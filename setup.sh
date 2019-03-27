#!bin/bash

echo INSTALLATION OF MLy PACKAGE

echo -e "The current installation path is: $PWD and it will create the file MLy \n" 
echo -e "Whould you like to change path? y/n"

read ch_path

while [ "$ch_path" != "n"  -a  "$ch_path" != "y" ]; do
    read ch_path
done
    
if [ "$ch_path" = "y" ]; then
    echo "Insert the path of installation (do not include the last '/'): "
    read the_path
    
    while [ ! -d "$the_path" ]; do
        echo -e "Path does not exist. \n Insert the path of installation: "
        read the_path
    done
elif [ "$ch_path" = "n" ]; then
    the_path=$PWD
    
fi

cd $the_path
chmod u+w .

mkdir MLy
cd MLy

mkdir trainings datasets ligo_data injections
    





