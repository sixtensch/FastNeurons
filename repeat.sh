echo "Repeating $1 times.";

for ((i = 0; i < $1; i++));
do
    "./$2"
    echo -n "."
done;

echo ""