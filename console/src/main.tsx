import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import enUS from 'antd/locale/en_US';
import './i18n';
import i18n from './i18n';
import App from './App';

function Root() {
  const [locale, setLocale] = useState(i18n.language === 'en' ? enUS : zhCN);

  useEffect(() => {
    const handleLangChange = (lng: string) => {
      setLocale(lng === 'en' ? enUS : zhCN);
    };
    i18n.on('languageChanged', handleLangChange);
    return () => { i18n.off('languageChanged', handleLangChange); };
  }, []);

  return (
    <ConfigProvider locale={locale}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ConfigProvider>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
