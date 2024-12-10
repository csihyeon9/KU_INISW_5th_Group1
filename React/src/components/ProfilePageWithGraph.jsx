// ProfilePageWithGraph.jsx
import GraphVisualization from './GraphVisualization';
import { useState } from 'react';
import { 
  Search, 
  Menu, 
  Bell, 
  Settings, 
  ChevronDown,
  Network,
  TrendingUp,
  ShoppingCart,
  Globe,
  Home,
  Building2,
  Factory,
  Wallet,
  LineChart,
  User,
  Users,
  HelpCircle,
  LogOut,
  Clock, 
  Bookmark,
} from 'lucide-react';
import CategorySection from '../hooks/CategorySection';
import ArticleSelection from '../hooks/ArticleSelection';
const ProfilePage = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [activeSection, setActiveSection] = useState('myGraph');
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [graphUpdateTrigger, setGraphUpdateTrigger] = useState(0);

  const articleItems = [
    { id: 'record1', name: '최근 본 기사', Icon: Clock },
    { id: 'record2', name: '스크랩한 기사', Icon: Bookmark },
  ];

  const menuItems = [
    { id: 'myGraph', name: 'My 지식그래프', Icon: Network },
    { id: 'general', name: '인과 or 상관성', Icon: TrendingUp },
    { id: 'life', name: '변화 & 추세', Icon: ShoppingCart },
    { id: 'global', name: '시장 및 거래', Icon: Globe },
    { id: 'realestate', name: '정책 & 제도', Icon: Home },
    { id: 'venture', name: '기업 활동', Icon: Building2 },
    { id: 'industry', name: '금융 상품 & 자산', Icon: Factory },
    { id: 'finance', name: '위험 및 위기', Icon: Wallet },
    { id: 'stock', name: '기술 및 혁신', Icon: LineChart },
  ];

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const toggleProfileMenu = () => {
    setIsProfileMenuOpen(!isProfileMenuOpen);
  };

  // 기사 삭제 시 그래프 업데이트 처리
  const handleArticleDelete = (articleId) => {
    setGraphUpdateTrigger(prev => prev + 1); // 그래프 업데이트 트리거
  };

  return (
    <div className="min-h-screen bg-[#f4f7fe]">
      {/* Top Navigation Bar */}
      <div className="fixed top-0 left-0 right-0 h-16 bg-white shadow-sm flex items-center justify-between px-6 z-50">
        <div className="flex items-center gap-4 cursor-pointer" onClick={toggleSidebar}>
          <Menu className="text-gray-500" />
        </div>
        <div className="flex items-center gap-4">
          <Bell className="text-gray-500 cursor-pointer" />
          <Settings className="text-gray-500 cursor-pointer" />
          <div className="relative">
            <div 
              className="flex items-center gap-2 cursor-pointer" 
              onClick={toggleProfileMenu}
            >
              <div className="w-8 h-8 rounded-full overflow-hidden">
                <img src="/user-profile.png" alt="Profile" className="w-full h-full object-cover" />
              </div>
              <ChevronDown className="w-4 h-4 text-gray-500" />
            </div>

            {/* Profile Dropdown Menu */}
            {isProfileMenuOpen && (
              <div className="absolute right-0 mt-2 w-60 bg-white rounded-lg shadow-lg py-2 z-50">
                <div className="px-4 py-3 border-b border-gray-100">
                  <div className="flex items-center gap-3">
                    <img src="/user-profile.png" alt="Profile" className="w-10 h-10 rounded-full" />
                    <div>
                      <div className="font-medium">Levi Ackerman</div>
                      <div className="text-sm text-gray-500">Survey Corps Special Operations Squad</div>
                    </div>
                  </div>
                </div>

                <div className="py-2">
                  <button className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3">
                    <User className="w-4 h-4 text-gray-500" />
                    <span>View Profile</span>
                  </button>
                  <button className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3">
                    <Settings className="w-4 h-4 text-gray-500" />
                    <span>Account Settings</span>
                  </button>
                  <button className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3">
                    <Bell className="w-4 h-4 text-gray-500" />
                    <span>Notifications</span>
                  </button>
                  <button className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3">
                    <Users className="w-4 h-4 text-gray-500" />
                    <span>Switch Account</span>
                  </button>
                  <button className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3">
                    <HelpCircle className="w-4 h-4 text-gray-500" />
                    <span>Help Center</span>
                  </button>
                  <div className="border-t border-gray-100 mt-2 pt-2">
                    <button className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3 text-red-500">
                      <LogOut className="w-4 h-4" />
                      <span>Logout</span>
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div className={`fixed left-0 top-16 bottom-0 w-64 bg-white shadow-sm p-4 transition-transform duration-300 ease-in-out overflow-y-auto ${
  isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
  }`}>
        <div className="space-y-6">
          {/* Profile section */}
          <div className="flex items-center gap-3 px-4">
            <img src="/user-profile.png" alt="Profile" className="w-10 h-10 rounded-full" />
            <div>
              <div className="font-semibold text-sm">USER_NAME</div>
              <div className="text-xs text-gray-500">Administrator</div>
            </div>
          </div>

          {/* Search box */}
          <div className="relative px-4">
            <input
              type="text"
              placeholder="키워드 입력..."
              className="w-full pl-8 pr-4 py-2 rounded-lg bg-[#f4f7fe] border-none outline-none"
            />
            <Search className="absolute left-6 top-2.5 w-4 h-4 text-gray-400" />
          </div>

          {/* Navigation menu */}
          <div className="space-y-6">
            {/* My 지식그래프 menu item */}
            <div
              onClick={() => setActiveSection('myGraph')}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg cursor-pointer ${
                activeSection === 'myGraph'
                  ? 'text-[#32CD32] bg-[#e8f5e9]'
                  : 'text-gray-600 hover:bg-[#e8f5e9]'
              }`}
            >
              <Network className="w-4 h-4" />
              <div className="text-sm font-medium">My 지식그래프</div>
            </div>

            {/* 기사 목록 section */}
              <ArticleSelection
                articleitems={articleItems}
                activeSection={activeSection}
                setActiveSection={(setActiveSection)}
                onArticleDelete={handleArticleDelete}
              />
            {/* 카테고리별 기사 section */}
              <CategorySection 
                menuItems={menuItems} 
                activeSection={activeSection} 
                setActiveSection={setActiveSection}
                onCategoryChange={setSelectedCategories}
              />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className={`transition-all duration-300 ease-in-out ${
        isSidebarOpen ? 'ml-64' : 'ml-0'
      } pt-16 p-6`}>
        {/* Header Card */}
        <div className="bg-white rounded-2xl shadow-sm p-6 mb-6 mt-5">
          
          {/* <div className="text-[#00c853] text-xl font-bold mb-4">Your Personal Knowledge Management Page</div>
          <div className="text-sm text-gray-600">
            뭐 쓰징
          </div> */}
        </div>
        {/* Graph Card */}
        <div className="bg-white rounded-2xl shadow-sm p-6">
          <GraphVisualization type={activeSection} updateTrigger={graphUpdateTrigger} />
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;