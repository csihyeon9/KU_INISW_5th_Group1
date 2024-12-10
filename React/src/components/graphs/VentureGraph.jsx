const VentureGraph = () => (
    <svg className="w-full h-[calc(100vh-400px)]" viewBox="0 0 400 300">
      <line x1="200" y1="50" x2="200" y2="250" stroke="#00c853" strokeWidth="2" />
      <line x1="100" y1="150" x2="300" y2="150" stroke="#00c853" strokeWidth="2" />
      <circle cx="200" cy="150" r="8" fill="#00c853" />
      <circle cx="200" cy="50" r="5" fill="#00c853" />
      <circle cx="200" cy="250" r="5" fill="#00c853" />
      <circle cx="100" cy="150" r="5" fill="#00c853" />
      <circle cx="300" cy="150" r="5" fill="#00c853" />
      <circle cx="150" cy="100" r="5" fill="#00c853" />
      <circle cx="250" cy="100" r="5" fill="#00c853" />
      <circle cx="150" cy="200" r="5" fill="#00c853" />
      <circle cx="250" cy="200" r="5" fill="#00c853" />
    </svg>
  );
  
  export default VentureGraph;