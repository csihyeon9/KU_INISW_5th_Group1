const RealEstateGraph = () => (
    <svg className="w-full h-[calc(100vh-400px)]" viewBox="0 0 400 300">
      <rect x="150" y="100" width="100" height="100" fill="none" stroke="#00c853" strokeWidth="2" />
      <line x1="150" y1="100" x2="100" y2="150" stroke="#00c853" strokeWidth="2" />
      <line x1="250" y1="100" x2="300" y2="150" stroke="#00c853" strokeWidth="2" />
      <line x1="150" y1="200" x2="100" y2="250" stroke="#00c853" strokeWidth="2" />
      <line x1="250" y1="200" x2="300" y2="250" stroke="#00c853" strokeWidth="2" />
      <circle cx="150" cy="100" r="5" fill="#00c853" />
      <circle cx="250" cy="100" r="5" fill="#00c853" />
      <circle cx="150" cy="200" r="5" fill="#00c853" />
      <circle cx="250" cy="200" r="5" fill="#00c853" />
      <circle cx="100" cy="150" r="5" fill="#00c853" />
      <circle cx="300" cy="150" r="5" fill="#00c853" />
      <circle cx="100" cy="250" r="5" fill="#00c853" />
      <circle cx="300" cy="250" r="5" fill="#00c853" />
    </svg>
  );
  
  export default RealEstateGraph;