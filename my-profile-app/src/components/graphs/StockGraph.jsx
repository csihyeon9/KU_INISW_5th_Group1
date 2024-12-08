const StockGraph = () => (
  <svg className="w-full h-[calc(100vh-400px)]" viewBox="0 0 400 300">
    <polyline 
      points="50,250 100,200 150,220 200,180 250,150 300,100 350,120" 
      fill="none" 
      stroke="#00c853" 
      strokeWidth="2"
    />
    <circle cx="50" cy="250" r="5" fill="#00c853" />
    <circle cx="100" cy="200" r="5" fill="#00c853" />
    <circle cx="150" cy="220" r="5" fill="#00c853" />
    <circle cx="200" cy="180" r="5" fill="#00c853" />
    <circle cx="250" cy="150" r="5" fill="#00c853" />
    <circle cx="300" cy="100" r="5" fill="#00c853" />
    <circle cx="350" cy="120" r="5" fill="#00c853" />
  </svg>
);

export default StockGraph;