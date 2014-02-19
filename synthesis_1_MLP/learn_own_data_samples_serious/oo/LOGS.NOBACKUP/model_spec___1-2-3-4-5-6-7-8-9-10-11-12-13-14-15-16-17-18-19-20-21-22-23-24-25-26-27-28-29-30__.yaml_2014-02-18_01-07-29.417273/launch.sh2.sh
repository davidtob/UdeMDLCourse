#!/bin/bash
export HOME=/u/belius

cd /data/lisatmp/ift6266h14/belius/synthesis_1_MLP/learn_own_data_samples_serious/oo
[ -r "LOGS.NOBACKUP/model_spec___1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29-30__.yaml_2014-02-18_01-07-29.417273/.bashrc" ];echo "Can read the source file? " $? 1>&2 
source LOGS.NOBACKUP/model_spec___1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29-30__.yaml_2014-02-18_01-07-29.417273/.bashrc
klist -5
echo "Executing on " `/bin/hostname` 1>&2
echo "HOSTNAME: ${HOSTNAME}" 1>&2
echo "PATH: $PATH" 1>&2
echo "PYTHONPATH: $PYTHONPATH" 1>&2
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" 1>&2
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS" 1>&2
echo "CONDOR_JOB_LOGDIR: $CONDOR_JOB_LOGDIR" 1>&2
echo "HOME: $HOME" 1>&2
grep -E "^GlobalJobId" $_CONDOR_JOB_AD 1>&2
pwd 1>&2
echo "nb args: $#" 1>&2
echo "Running: command: \"$@\"" 1>&2
echo "ulimit cmd: ulimit -v 1484800"
ulimit -v 1484800
[ -x "$1" ];echo "Can execute the cmd? " $? 1>&2
sh -c "$@"
ret=$?
rm -f echo ${KRB5CCNAME:5}
echo "return value ${ret}"
exit ${ret}
