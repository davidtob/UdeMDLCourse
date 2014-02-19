. ~/.profile
if [ -e "/opt/lisa/os/.local.bashrc" ];then source /opt/lisa/os/.local.bashrc; else source /data/lisa/data/local_export/.local.bashrc; fi

export PYTHONPATH=$PYTHONPATH:/data/lisatmp/ift6266h14/belius/lib/python2.7/site-packages:/data/lisatmp/ift6266h14/belius/lib/python2.7/site-packages
export PYLEARN2_DATA_PATH=/data/lisa/data
