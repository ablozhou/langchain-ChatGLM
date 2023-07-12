
cd ./service
nohup npm start > service.log &
echo "Start service complete!"


cd ..
echo "" > front.log
nohup npm run dev > front.log &
echo "Start front complete!"
tail -f front.log
