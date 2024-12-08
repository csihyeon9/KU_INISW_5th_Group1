import MyGraph from './graphs/MyGraph';
import GeneralGraph from './graphs/GeneralGraph';
import LifeGraph from './graphs/LifeGraph';
import GlobalGraph from './graphs/GlobalGraph';
import RealEstateGraph from './graphs/RealEstateGraph';
import VentureGraph from './graphs/VentureGraph';
import IndustryGraph from './graphs/IndustryGraph';
import FinanceGraph from './graphs/FinanceGraph';
import StockGraph from './graphs/StockGraph';

const GraphVisualization = ({ type }) => {
  const graphs = {
    myGraph: <MyGraph />,
    general: <GeneralGraph />,
    life: <LifeGraph />,
    global: <GlobalGraph />,
    realestate: <RealEstateGraph />,
    venture: <VentureGraph />,
    industry: <IndustryGraph />,
    finance: <FinanceGraph />,
    stock: <StockGraph />,
  };

  return graphs[type] || graphs.myGraph;
};

export default GraphVisualization;