for i in {1..100}; do
    echo "Iteration $i"
    python demo.py --mode sweep-carbs --vec native --env gpudrive --track
done
