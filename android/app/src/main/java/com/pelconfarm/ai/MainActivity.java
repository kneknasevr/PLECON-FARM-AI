package com.pelconfarm.ai;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.CookieManager;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

public class MainActivity extends Activity {
 private static final String HOME = "https://pelcon-farm-ai.streamlit.app/";
 private static final int FILE_REQUEST = 4102;
 private WebView webView;
 private ValueCallback<Uri[]> uploadCallback;
 @Override public void onCreate(Bundle state) {
  super.onCreate(state);
  webView = new WebView(this);
  setContentView(webView);
  WebSettings settings = webView.getSettings();
  settings.setJavaScriptEnabled(true);
  settings.setDomStorageEnabled(true);
  settings.setAllowFileAccess(false);
  settings.setAllowContentAccess(true);
  settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
  CookieManager.getInstance().setAcceptCookie(true);
  webView.setWebViewClient(new WebViewClient() {
   @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
    Uri uri = request.getUrl();
    if ("https".equals(uri.getScheme()) && "pelcon-farm-ai.streamlit.app".equals(uri.getHost())) return false;
    try { startActivity(new Intent(Intent.ACTION_VIEW, uri)); }
    catch (ActivityNotFoundException e) { Toast.makeText(MainActivity.this,"Δεν βρέθηκε εφαρμογή για τον σύνδεσμο",Toast.LENGTH_SHORT).show(); }
    return true;
   }
  });
  webView.setWebChromeClient(new WebChromeClient() {
   @Override public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
    if(uploadCallback!=null) uploadCallback.onReceiveValue(null);
    uploadCallback=callback;
    try { startActivityForResult(params.createIntent(),FILE_REQUEST); }
    catch (ActivityNotFoundException e) { uploadCallback=null; callback.onReceiveValue(null); return false; }
    return true;
   }
  });
  webView.setDownloadListener((url, agent, disposition, mime, length) -> {
   try { startActivity(new Intent(Intent.ACTION_VIEW,Uri.parse(url))); }
   catch (Exception e) { Toast.makeText(this,"Αδυναμία ανοίγματος λήψης",Toast.LENGTH_SHORT).show(); }
  });
  webView.loadUrl(HOME);
 }
 @Override protected void onActivityResult(int request,int result,Intent data) {
  super.onActivityResult(request,result,data);
  if(request!=FILE_REQUEST || uploadCallback==null) return;
  Uri[] files=result==RESULT_OK?WebChromeClient.FileChooserParams.parseResult(result,data):null;
  uploadCallback.onReceiveValue(files);
  uploadCallback=null;
 }
 @Override public void onBackPressed() { if(webView.canGoBack()) webView.goBack(); else super.onBackPressed(); }
 @Override protected void onDestroy() { if(uploadCallback!=null) uploadCallback.onReceiveValue(null); webView.destroy(); super.onDestroy(); }
}
