function remkdir(folder_name)
% remkdir - Create a folder if it doesn't exist

if ~exist(folder_name,'dir')
    mkdir(folder_name);
end