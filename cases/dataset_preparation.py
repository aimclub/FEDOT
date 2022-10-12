import os
import tarfile


def unpack_archived_data(archive_name: str):
    archive_path = os.path.abspath(f'{archive_name}.tar.gz')
    if (os.path.exists(archive_path) and
            os.path.basename(archive_name) not in os.listdir(os.path.dirname(archive_path))):
        with tarfile.open(archive_path) as file:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(file, path=os.path.dirname(archive_path))
        print('Unpacking finished')
    else:
        print('Archive already unpacked')
