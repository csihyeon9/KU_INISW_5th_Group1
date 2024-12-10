// ArticleSelection.jsx
import { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { ChevronDown, ChevronRight, Clock, Bookmark, Trash2 } from 'lucide-react';

const ArticleSelection = ({ 
  activeSection, 
  setActiveSection, 
  onArticleDelete // 새로운 prop 추가
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [dynamicMenuItems, setDynamicMenuItems] = useState([]);
  
  const staticItems = [
    { id: 'record1', name: '최근 본 기사', Icon: Clock },
    { id: 'record2', name: '스크랩한 기사', Icon: Bookmark }
  ];

  useEffect(() => {
    const fetchDynamicItems = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/articles');
        const articles = await response.json();
        
        const items = articles.map(article => ({
          id: article._id,
          name: article.title,
          Icon: ChevronRight,
          isDynamic: true
        }));
        
        setDynamicMenuItems(items);
      } catch (error) {
        console.error('Error fetching menu items:', error);
        setDynamicMenuItems([]);
      }
    };

    fetchDynamicItems();
  }, []);

  const deleteArticle = async (articleId, e) => {
    e.stopPropagation();
    
    try {
      const response = await fetch(`http://127.0.0.1:8000/articles/${articleId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // UI 업데이트
      setDynamicMenuItems((prevItems) => prevItems.filter(item => item.id !== articleId));
      
      // 선택된 항목이 삭제된 경우 선택 해제
      if (activeSection === articleId) {
        setActiveSection(null);
      }

      // 그래프 업데이트를 위해 상위 컴포넌트에 알림
      onArticleDelete(articleId);

    } catch (error) {
      console.error('Error deleting article:', error);
    }
  };

  const allItems = [...staticItems, ...dynamicMenuItems];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between px-4">
        <h2 className="text-sm font-semibold text-gray-800">기사 목록</h2>
        <button 
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-gray-600 hover:bg-gray-100 rounded-full p-1"
          aria-label={isExpanded ? "Collapse article list" : "Expand article list"}
        >
          {isExpanded ? 
            <ChevronDown className="w-5 h-5" /> : 
            <ChevronRight className="w-5 h-5" />
          }
        </button>
      </div>

      {isExpanded && (
        <div className="space-y-1">
          {allItems.map(item => (
            <div
              key={item.id}
              className={`flex items-center justify-between px-4 py-3 rounded-lg cursor-pointer transition-colors ${
                activeSection === item.id
                  ? 'text-[#00c853] bg-[#e8f5e9]'
                  : 'text-gray-600 hover:bg-[#e8f5e9]'
              }`}
              role="button"
              tabIndex={0}
              onKeyPress={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  setActiveSection(item.id);
                }
              }}
            >
              <div 
                onClick={() => setActiveSection(item.id)}
                className="flex items-center gap-3 flex-grow"
              >
                <item.Icon className="w-4 h-4" />
                <div className="text-sm font-medium">{item.name}</div>
              </div>
              
              {item.isDynamic && (
                <button
                  onClick={(e) => deleteArticle(item.id, e)}
                  className="text-gray-400 hover:text-red-600 p-1 rounded transition-colors"
                  aria-label={`Delete ${item.name}`}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

ArticleSelection.propTypes = {
  activeSection: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  setActiveSection: PropTypes.func.isRequired,
  onArticleDelete: PropTypes.func.isRequired
};

export default ArticleSelection;