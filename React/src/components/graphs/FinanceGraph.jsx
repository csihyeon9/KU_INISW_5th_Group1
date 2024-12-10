const FinanceGraph = () => (
    <svg className="w-full h-[calc(100vh-400px)]" viewBox="0 0 400 300">
      <circle cx="200" cy="150" r="80" stroke="#00c853" strokeWidth="2" fill="none" />
      <line x1="140" y1="150" x2="260" y2="150" stroke="#00c853" strokeWidth="2" />
      <line x1="200" y1="90" x2="200" y2="210" stroke="#00c853" strokeWidth="2" />
      <circle cx="200" cy="150" r="8" fill="#00c853" />
      <circle cx="140" cy="150" r="5" fill="#00c853" />
      <circle cx="260" cy="150" r="5" fill="#00c853" />
      <circle cx="200" cy="90" r="5" fill="#00c853" />
      <circle cx="200" cy="210" r="5" fill="#00c853" />
    </svg>
  );
  
  export default FinanceGraph;