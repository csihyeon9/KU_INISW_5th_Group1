import { useState } from 'react';
import PropTypes from 'prop-types';
import { ChevronDown, ChevronRight, Check } from 'lucide-react';

const CategorySection = ({ 
  menuItems, 
  setActiveSection,
  onCategoryChange 
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [selectedCategories, setSelectedCategories] = useState([]);

  const handleCategoryToggle = (categoryId) => {
    const currentIndex = selectedCategories.indexOf(categoryId);
    const newSelectedCategories = [...selectedCategories];

    if (currentIndex === -1) {
      newSelectedCategories.push(categoryId);
    } else {
      newSelectedCategories.splice(currentIndex, 1);
    }

    setSelectedCategories(newSelectedCategories);
    
    if (onCategoryChange) {
      onCategoryChange(newSelectedCategories);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between px-4">
        <h2 className="text-sm font-semibold text-gray-800">카테고리별 기사</h2>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-gray-600 hover:bg-gray-100 rounded-full p-1"
        >
          {isExpanded ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
        </button>
      </div>

      {isExpanded && (
        <div className="space-y-1">
          {menuItems.slice(1).map(item => (
            <div
              key={item.id}
              className={`relative flex items-center px-4 py-3 rounded-lg cursor-pointer group ${
                selectedCategories.includes(item.id)
                  ? 'bg-[#e8f5e9]'
                  : 'hover:bg-gray-100'
              }`}
              onClick={() => {
                handleCategoryToggle(item.id);
                setActiveSection(item.id);
              }}
            >
              {/* Left side - Icon and Name */}
              <div className="flex items-center gap-3 flex-grow">
                <item.Icon className="w-4 h-4 text-gray-600" />
                <div className="text-sm font-medium text-gray-800">{item.name}</div>
              </div>

              {/* Right side - Checkbox */}
              <div 
                className={`w-5 h-5 border rounded transition-all duration-200 flex items-center justify-center ${
                  selectedCategories.includes(item.id)
                    ? 'bg-[#00c853] border-[#00c853]'
                    : 'border-gray-300 bg-white'
                }`}
              >
                {selectedCategories.includes(item.id) && (
                  <Check className="w-4 h-4 text-white" />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

CategorySection.propTypes = {
  menuItems: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      name: PropTypes.string.isRequired,
      Icon: PropTypes.elementType.isRequired
    })
  ).isRequired,
  activeSection: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  setActiveSection: PropTypes.func.isRequired,
  onCategoryChange: PropTypes.func
};

CategorySection.defaultProps = {
  onCategoryChange: () => {}
};

export default CategorySection;