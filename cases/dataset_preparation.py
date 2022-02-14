import os
import tarfile


def unpack_archived_data(archive_name: str):
    archive_path = os.path.abspath(f'{archive_name}.tar.gz')
    if (os.path.exists(archive_path) and
            os.path.basename(archive_name) not in os.listdir(os.path.dirname(archive_path))):
        with tarfile.open(archive_path) as file:
            file.extractall(path=os.path.dirname(archive_path))
        print('Unpacking finished')
    else:
        print('Archive already unpacked')
