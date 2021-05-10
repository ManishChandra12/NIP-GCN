git clone https://github.com/snap-stanford/snap.git
cd snap
make -C snap-core
make -C examples/node2vec
cp examples/node2vec/node2vec ../node2vec_cpp/
cd ..
rm -rf snap
pip install -r requirements.txt