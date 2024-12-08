const LifeGraph = () => (
    <svg className="w-full h-[calc(100vh-400px)]" viewBox="0 0 400 300">
      <circle cx="200" cy="150" r="100" stroke="#00c853" strokeWidth="2" fill="none" />
      <circle cx="200" cy="50" r="5" fill="#00c853" />
      <circle cx="300" cy="150" r="5" fill="#00c853" />
      <circle cx="200" cy="250" r="5" fill="#00c853" />
      <circle cx="100" cy="150" r="5" fill="#00c853" />
      <line x1="200" y1="50" x2="300" y2="150" stroke="#00c853" strokeWidth="1" />
      <line x1="300" y1="150" x2="200" y2="250" stroke="#00c853" strokeWidth="1" />
      <line x1="200" y1="250" x2="100" y2="150" stroke="#00c853" strokeWidth="1" />
      <line x1="100" y1="150" x2="200" y2="50" stroke="#00c853" strokeWidth="1" />
    </svg>
  );
  
  export default LifeGraph;