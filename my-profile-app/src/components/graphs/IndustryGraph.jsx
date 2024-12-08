const IndustryGraph = () => (
    <svg className="w-full h-[calc(100vh-400px)]" viewBox="0 0 400 300">
      <line x1="50" y1="150" x2="350" y2="150" stroke="#00c853" strokeWidth="2" />
      <circle cx="50" cy="150" r="5" fill="#00c853" />
      <circle cx="150" cy="150" r="5" fill="#00c853" />
      <circle cx="250" cy="150" r="5" fill="#00c853" />
      <circle cx="350" cy="150" r="5" fill="#00c853" />
      <line x1="150" y1="150" x2="150" y2="100" stroke="#00c853" strokeWidth="2" />
      <line x1="250" y1="150" x2="250" y2="100" stroke="#00c853" strokeWidth="2" />
      <circle cx="150" cy="100" r="5" fill="#00c853" />
      <circle cx="250" cy="100" r="5" fill="#00c853" />
      <line x1="150" y1="150" x2="150" y2="200" stroke="#00c853" strokeWidth="2" />
      <line x1="250" y1="150" x2="250" y2="200" stroke="#00c853" strokeWidth="2" />
      <circle cx="150" cy="200" r="5" fill="#00c853" />
      <circle cx="250" cy="200" r="5" fill="#00c853" />
    </svg>
  );
  
  export default IndustryGraph;