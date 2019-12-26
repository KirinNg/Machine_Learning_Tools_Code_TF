from fabric.tasks import task
from fabric import Connection
from static_file import FILE_UPLOAD_MANAGER
import os
from invoke import Context
import threading
from tqdm import tqdm
import qiniu
from invoke import Context
STATIC_DIR = os.curdir + '/dist/spa'
FILE_LIST = []
PROJECT_NAME = 'bubble-admin'
local_con = Context()


def walk_static_dir(dir_path):
    if not os.path.isdir(dir_path):
        return
    files = os.listdir(dir_path)
    try:
        for file in files:
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                key = PROJECT_NAME + \
                    file_path.replace(STATIC_DIR, '').replace(
                        '\\', '/').replace('\\\\', '/')
                FILE_LIST.append({
                    'file_path': file_path,
                    'key': key
                })
            elif os.path.isdir(file_path):
                walk_static_dir(file_path)
    except Exception as e:
        raise e


def upload_file(upload_token: str, key: str, file_path: str, progress_bar: tqdm = None):
    qiniu.put_file(upload_token, key, file_path)
    if progress_bar is not None:
        progress_bar.update()


@task
def upload_dist_files(_):
    FILE_UPLOAD_MANAGER.delete_file_list(PROJECT_NAME)
    print('Old files deleted')
    walk_static_dir(STATIC_DIR)
    progress_bar = tqdm(desc='Uploading', total=len(FILE_LIST))
    upload_token = FILE_UPLOAD_MANAGER.get_upload_token()
    threads = [
        threading.Thread(target=upload_file, args=(
            upload_token, file['key'], file['file_path'], progress_bar))
        for file in FILE_LIST
    ]
    for t in threads:
        # t.setDaemon(True)
        t.start()
    for j in threads:
        j.join()


@task
def deploy(_):
    c = Connection('39.107.82.164', user='root', connect_kwargs={
        'password': 'Rick_zhang_168!'
    })
    ctx = Context()
    ctx.run('npm run-script build')
    c.put(f'{STATIC_DIR}/index.html', remote=f'/data/{PROJECT_NAME}/')
    upload_dist_files(_)


def commit(message):
    local_con.run('git add -A && git commit -m ' + message, warn=True)


def pull():
    local_con.run('git pull', warn=True)


def push():
    local_con.run('git push', warn=True)


@task
def git(ctx, message=''):
    commit(message)
    pull()
    push()