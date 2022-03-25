import sys
import time
from subprocess import check_output, CalledProcessError, STDOUT
import os
import csv
import threading
from argparse import ArgumentParser
from pathlib import Path
from typing import List

MAX_THREAD_NUM = 8


def remove_grade(s):
    s = s[:s.rfind('\n')]
    return s[:s.rfind('\n')] + '\n'


class Mythread(threading.Thread):
    thread_pool = threading.BoundedSemaphore(MAX_THREAD_NUM)

    def __init__(self, sid):
        super(Mythread, self).__init__()
        self.sid = sid
        self.score = None
        self.script = self.get_base_script()
        self.log = 'code not found, name error or runtime error\n'

    def run(self):
        self.thread_pool.acquire()
        print('Executing student {}\'s code...'.format(self.sid))
        log_path = args.log_dir / '{}.txt'.format(self.sid)
        log_file = log_path.open('w')
        self.script += ['autograder.py', '-S', '{}/{}_hw2.py'.format(args.code_dir, self.sid)]
        self.get_log_and_score()
        log_file.write(self.log)
        log_file.close()
        print('Finish grading student {}.'.format(self.sid))
        self.thread_pool.release()

    def join(self, *args_) -> List[str]:
        threading.Thread.join(self)
        return self.score

    def get_log_and_score(self):
        # print threading.current_thread().getName() + '\n'
        try:
            out = check_output(self.script, stderr=STDOUT)
            if isinstance(out, bytes):
                out = out.decode('utf-8')
            self.score = out.splitlines()[-1].split(',')
            total = sum(map(int, self.score[1:]))
            self.score.append(str(total))
            self.log = remove_grade(out)
        except CalledProcessError as e:
            self.score = [self.sid] + ['0'] * 4 + ['-1']
            log = e.output
            if isinstance(log, bytes):
                self.log = log.decode('utf-8')

    def get_base_script(self):
        is_win = sys.platform.startswith('win')
        if is_win:
            script = ['py', '-2']
        else:
            script = ['python2']
        return script


def parallel_grading(student_list):
    thread_list = []
    sorted_unique_student_list = sorted(list(set(student_list)))
    for student_id in sorted_unique_student_list:
        student_id = student_id.strip()
        if student_id:
            t = Mythread(student_id)
            t.setDaemon(True)
            thread_list.append(t)
            t.start()
    return thread_list


def init_csv(csv_name):
    if not os.path.isfile(csv_name):
        return [['Student ID', 'Problem1', 'Problem2', 'Problem3', 'Problem4', 'Total Points']]
    with csv_name.open('r') as csv_file:
        return list(csv.reader(csv_file))


def main():
    csv_rows = init_csv(args.grade_path)
    if args.ID_list:
        id_list = args.ID_list
    else:
        with open(args.list_path, 'r') as student_list_file:
            id_list = student_list_file.read().splitlines()
    thread_list = parallel_grading(id_list)
    if not args.modify_grade:
        csv_rows = csv_rows[:1]
    for thread in thread_list:
        if args.modify_grade:
            grade_score: List[str] = thread.join()
            grade_exist = False
            for row in csv_rows[1:]:
                if row[0] == grade_score[0]:
                    grade_exist = True
                    print('Student {} :'.format(row[0]))
                    print('Old grade : ' + ', '.join(row))
                    print('New grade : ' + ', '.join(grade_score))
                    if int(grade_score[-1]) > int(row[-1]):
                        print('Update {}\'s total grade from {}  to {}'.format(row[0], row[-1], grade_score[-1]))
                        row[1:] = grade_score[1:]
                    else:
                        print('New grade is equal or lower than old grade')
            if not grade_exist:
                print('Append {}\'s grade : '.format(grade_score[0]))
                print(', '.join(grade_score))
                csv_rows.append(grade_score)
            print()
        else:
            grade_score = thread.join()
            csv_rows.append(grade_score)
    with args.grade_path.open('w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-C',
        "--code_dir",
        type=Path,
        help="Directory to Student Code.",
        default="./student_code",
    )
    parser.add_argument(
        '-IL',
        "--list_path",
        type=Path,
        help="Path to Student List.",
        default="./student_list.txt",
    )
    parser.add_argument(
        '-I',
        "--ID_list",
        type=str,
        nargs='+',
        help="Student ID List.",
        default=[]
    )
    parser.add_argument(
        '-L',
        "--log_dir",
        type=Path,
        help="Directory to save the log file.",
        default="./student_log",
    )
    parser.add_argument(
        '-G',
        "--grade_path",
        type=Path,
        help="Path to Grade file.",
        default="./grade.csv",
    )
    parser.add_argument(
        '-M',
        "--modify_grade",
        action='store_true',
        help="modify exist csv or create csv",
        default=False,
    )
    args_ = parser.parse_args()
    args_.log_dir.mkdir(parents=True, exist_ok=True)
    return args_


if __name__ == '__main__':
    args = parse_args()
    main()
