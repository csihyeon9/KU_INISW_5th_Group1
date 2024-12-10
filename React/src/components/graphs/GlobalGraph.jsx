const GlobalGraph = () => (
  <svg className="w-full h-[calc(100vh-400px)]" viewBox="0 0 400 300">
    <line x1="200" y1="150" x2="300" y2="100" stroke="#00c853" strokeWidth="2" />
    <line x1="300" y1="100" x2="350" y2="200" stroke="#00c853" strokeWidth="2" />
    <line x1="350" y1="200" x2="250" y2="250" stroke="#00c853" strokeWidth="2" />
    <line x1="250" y1="250" x2="150" y2="200" stroke="#00c853" strokeWidth="2" />
    <line x1="150" y1="200" x2="200" y2="150" stroke="#00c853" strokeWidth="2" />
    <circle cx="200" cy="150" r="8" fill="#00c853" />
    <circle cx="300" cy="100" r="5" fill="#00c853" />
    <circle cx="350" cy="200" r="5" fill="#00c853" />
    <circle cx="250" cy="250" r="5" fill="#00c853" />
    <circle cx="150" cy="200" r="5" fill="#00c853" />
  </svg>
);

export default GlobalGraph;