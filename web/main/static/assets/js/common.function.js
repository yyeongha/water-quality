jQuery(function($) {
	// jquery appendVal plugin
	$.fn.appendVal = function (newPart) {
		var result = this.each(function(){ 
			if( null != $(this).val() && "" != $(this).val() ){
				$(this).val( $(this).val() +","+ newPart );
			}else{
				$(this).val( $(this).val() + newPart); 
			}
		});
		return result;
	};

	$(".h2_arrow").on("click", function(e){
		history.go(-1);
		return false;
	});
});

// 레이어 팝업 닫힘
function layer_pop_close() {
	$(".layer_pop").removeAttr("style");
	$("#fade").hide();
	$(".layer_pop").hide();	
}

// 모바일 전화 걸기 
function fnCallTel(tel) {
	if (isMobile)	location.href="tel:" + tel;
}
// 모바일 이미지 확대 보기 
function fnMobileImageView(obj){
	if (isMobile) {
		var imgPath = $(obj).find("img").attr("src");
		targetLocation("/kr/popup/imageView.do?imgPath=" + imgPath);
	}
}
// 상단 기술 자료 검색 
function fnTopSearch(){
	var searchTopKeyword = $("input[name=searchTopKeyword]");
	if ( searchTopKeyword.val() == '') {
		alert("제품명을 입력하세요.");
		searchTopKeyword.focus();
		return;
	}

	location.href = encodeURI("/kr/information/support/search.do?searchCondition=10&searchKeyword=" + searchTopKeyword.val());
}


// msds 목록 
function getMsdsList(proCd, p) {
	//var jepumCode = proCd.substring(0, 4);
	var jepumCode = proCd;
    var params = "jepumCode=" + jepumCode + "&pageIndex=" + p;
    $.ajax({
        type: "get", 
        url : "/kr/cmmn/products/ajaxMsdsList.do",
        dataType : "html",  
        data : params,  
        cache:false,
        error:function ( e ){	                 
        },
        success:function ( html, status ){
			$(".layer_pop").html(html);
        },
        complete:function(xhr, status){
        	layer_pop_open();
        }
        
    });  
}
// 제품 상세 보기 
function getItemView($obj) {
		
	var seq = $obj.find("a.more_cont").attr("id");	    	
    var params = "seq=" + seq;
    $.ajax({
        type: "get", 
        url : "/kr/cmmn/products/ajaxItemView.do",
        dataType : "html",  
        data : params,
        cache:false,
        error:function ( e ){	                 
        },
        success:function ( html, status ){
			$obj.next().find("#detailInfo").html(html);
        },
        complete:function(xhr, status){
        	if($obj.next().css("visibility") == "visible") {
        		var catePath = $obj.next().find("#cateDepth").html();
        		$obj.next().find("dd.path").html(catePath);
        		productInfo.viewOn($obj);
        	} 
        }
        
    });  
}

//제품 상세 보기 레이어 팝업 
function getItemViewLayerPop(lnkUrl) {			
	if ( lnkUrl.length < 5 ) return;		
	var title = "제품 상세";
    var params = "partSeq=0";
    $.ajax({
        type: "get", 
        url : lnkUrl,
        dataType : "html",  
        data : params,
        cache:false,
        error:function ( e ){	                 
        },
        success:function ( html, status ){
        	$(".layer_pop").html(html);
        },
        complete:function(xhr, status){
        	layer_pop_open();
        	if ( title != '' ) $("#layerTitle").text(title);
        }	        
    });  
}	

// 상세 인쇄 
var fnPrint = (function (obj){	
	$(obj).parent().parent().printThis({
		debug: false,
		loadCSS: "/front/kr/asset/css/defaultPrint.css"
	});
});  
// 제품 상세 팝업 인쇄
var fnPrint2 = (function (obj){	
	$(obj).parent().parent().parent().printThis({
		debug: false,
		loadCSS: "/front/kr/asset/css/defaultPrint.css"
	});
}); 


//다음카카오 지도 api 호출 ( 사업장, 대리점 찾기 )
function getMapApi(objId, addr){
	if ( $("#" + objId).parent().hasClass("on") || $("#" + objId).parent().hasClass("eduCenMap") ) {
		var mapMarkerImg = "/front/kr/asset/images/company/map_point_m_company.png";
		if (objId.indexOf("mapAgent") > - 1) mapMarkerImg = "/front/kr/asset/images/company/map_point_m_agent.png";
		
		var sangho = $("#" + objId).prev().find("#sangho").text();
		console.log(sangho);
		var mapContainer = document.getElementById(objId), // 지도를 표시할 div 
	    mapOption = {  
	        center: new daum.maps.LatLng(33.450701, 126.570667), // 지도의 중심좌표
	        level: 3 // 지도의 확대 레벨
	    };
		// 지도를 생성합니다    
		var map = new daum.maps.Map(mapContainer, mapOption);

//			var imageSrc = mapMarkerImg, // 마커이미지의 주소입니다    
//		    imageSize = new naver.maps.Size(64, 69), // 마커이미지의 크기입니다
//		    imageOption = {offset: new naver.maps.Point(27, 69)}; // 마커이미지의 옵션입니다. 마커의 좌표와 일치시킬 이미지 안에서의 좌표를 설정합니다.			      
//			// 마커의 이미지정보를 가지고 있는 마커이미지를 생성합니다
//			var markerImage = new naver.maps.MarkerImage(imageSrc, imageSize, imageOption);
	    
		// 주소-좌표 변환 객체를 생성합니다
		var geocoder = new daum.maps.services.Geocoder();
		geocoder.addressSearch(addr, function(result, status) {
			if (status === daum.maps.services.Status.OK) {
		        var coords = new daum.maps.LatLng(result[0].y, result[0].x);
		        var imageSize = new daum.maps.Size(191, 64), // 마커이미지의 크기입니다
	                imageOption = {offset: new daum.maps.Point(27, 69)}; // 마커이미지의 옵션입니다. 마커의 좌표와 일치시킬 이미지 안에서의 좌표를 설정합니다.
	          
			    // 마커의 이미지정보를 가지고 있는 마커이미지를 생성합니다
			    var markerImage = new daum.maps.MarkerImage(mapMarkerImg, imageSize, imageOption);
	
			    // 마커를 생성합니다
			    var marker = new daum.maps.Marker({
		            map: map,
		            position: coords,
			        image: markerImage // 마커이미지 설정 
			    });
	
		        // 지도의 중심을 결과값으로 받은 위치로 이동시킵니다
		        map.setCenter(coords);

		        // 지도 확대 축소를 제어할 수 있는  줌 컨트롤을 생성합니다
				var zoomControl = new daum.maps.ZoomControl();
				map.addControl(zoomControl, daum.maps.ControlPosition.RIGHT);
	    	}
		});	  	    		
	}else{
		setTimeout("getMapApi('" + objId + "', '" + addr + "');", 500);  
	}

}

  
// 사업장 대리점 주소 복사 
var fnAddrCopy = (function(addr){		
	
	//var url = location.href;
	var txt = addr;
	var bResult = false;
	
	try {
	if( window.clipboardData ) {
		bResult = window.clipboardData.setData("Text", txt);
	}else{
		temp = prompt("아래의 URL을 복사(Ctrl+C)하여\n원하는 곳에 붙여넣기(Ctrl+V)하세요.", txt);
	}
	
	if( bResult == true) {
		alert("게시물 주소가 복사되었습니다.\nCtrl+V 로 붙여넣기 할 수 있습니다.");  
		return;
	}  
	}catch(e){		
	}
	
});

//############메인 공지 팝업################################
var _MainPopup = new Array();
  
var addPopup = (function(idx, title) {
	 if(getCookie("jeviscoPop"+ idx) != "jeviscoPop"+ idx +"_cookie") {
		var conts = $("#mainPopupConts" + idx).html();		
		var obj = new Object();
		obj.IDX = idx;		  
		obj.title	= title;  
		obj.content = conts;
		_MainPopup.push(obj);
	 }
});

var showDefaultPopup = (function () {  
	//try {
		var TmpArr = _MainPopup.reverse();
		for(var i = 0; i < TmpArr.length; i++) {
			var PopObj = TmpArr[i];				
		    //alert(document.cookie);			
			 if(getCookie("jeviscoPop"+PopObj.IDX) != "jeviscoPop"+PopObj.IDX+"_cookie") {
			    var popup = new jQuery.Popup({
			        popupId : 'jeviscoPop' + PopObj.IDX,
			        popupTitle : PopObj.title,
			        popupMove: false,
			        popupBackground : true, 
			        popupHtml : PopObj.content,					
			    }); 				 	
				popup.openPopup();
			}
		}
	//} catch(e) {}
});
var fnPopClose = ( function(id){
	new jQuery.Popup().closePopup(id); 
	
});
// ############팝업 열기################################
function popWin(url, w, h, scroll, name) {
	var option = "status=no,height=" + h + ",width=" + w
			+ ",resizable=no,left=0,top=0,screenX=0,screenY=0,scrollbars="
			+ scroll;

	commonPopWin = window.open(url, name, option);
	commonPopWin.focus();
}

function confirmPopWin(url, w, h, scroll, name) {
	if (confirm("새 창으로 열립니다. 여시겠습니까?") == false)
		return;

	var option = "status=no,height=" + h + ",width=" + w
			+ ",resizable=no,left=0,top=0,screenX=0,screenY=0,scrollbars="
			+ scroll;

	commonPopWin = window.open(url, name, option);
	commonPopWin.focus();
}
function confirmTargetLocation(url) {
	if (confirm("새 창으로 열립니다. 여시겠습니까?") == false)
		return;
	var popWin = window.open('about:blank');
	popWin.location.href = url;
}
function targetLocation(url) {
	var popWin = window.open('about:blank');
	popWin.location.href = url;
}

function getQuerystring(key, default_){
  if (default_==null) default_=""; 
  key = key.replace(/[\[]/,"\\\[").replace(/[\]]/,"\\\]");
  var regex = new RegExp("[\\?&]"+key+"=([^&#]*)");
  var qs = regex.exec(window.location.href);
  if(qs == null)
    return "";  
  else
    return qs[1];
}

// 아이프레임 리사이즈
function fnIframeResize(_objId) {
	var iframeObj = document.getElementById(_objId);

	var childFrameHeight = iframeObj.contentWindow.document.body.offsetHeight;

	$('#' + _objId).css('height', childFrameHeight);

}

/** XSS 치환 return **/
function fnStripTag(str){  
	str = str.replace(/<[^<|>]*>|&nbsp;|\r\n/gi, "").trim();
	return str;
}

//페이징 이동 
function fnPaging(pageIndex){
    var frm = document.searchForm;
    $("input[name='pageIndex']").val( pageIndex );
    frm.submit();
}



//페이징 Url 이동 
function fnPagingUrl(pageIndex, _url){
    var frm = document.searchForm;
    $("input[name='pageIndex']").val( pageIndex );
    frm.action = _url;
    frm.submit();
}

//페이지 이동
function fnPage(_url) {
    var frm = document.searchForm;
    frm.action = _url;
    frm.method = "get";
    frm.submit();
}



//상세페이지 이동 
function fnView(_url, _seq) {
    var frm = document.searchForm;
	$("input[name='seq']").val( _seq );
	frm.action = _url;
	frm.method = "get";
	frm.submit();
}

// 첨부파일 다운로드
function downloadFile(obj, path, name) {
	if (isMobile){		
		$(obj).attr("href", path);
	}else{
		var url = encodeURI("/download.do?path=" + path + "&fileName=" + name);
		$("#ifr").attr("src", url);
		return false;
	}
}


function replaceAll(inputString, targetString, replacement){
	  var v_ret = null;
	  var v_regExp = new RegExp(targetString, "g");
	  v_ret = inputString.replace(v_regExp, replacement);
	  return v_ret;
}

//쿠키이용
var setCookie = (function(strName, strValue, days) {
	var objExpireDate = new Date();

	objExpireDate.setDate(objExpireDate.getDate() + days); 
	
	document.cookie = strName + "=" + escape(strValue) + "; path=/; expires=" + objExpireDate.toGMTString() + ";";

	//alert(document.cookie);
});

var setCookie2 = (function(strName, strValue, day) {  
	var objExpireDate = new Date();
	day = day || 1;
	objExpireDate.setDate(objExpireDate.getDate() + (24 * 60 * 60 * 1000 * day)); 
	
	document.cookie = strName + "=" + escape(strValue) + "; path=/; expires=" + objExpireDate.toGMTString() + ";";

	//alert(document.cookie);
});

var delCookie = (function(strName) {
	var objExpireDate = new Date();

	objExpireDate.setDate(objExpireDate.getDate() - 1);
	
	document.cookie = strName + "=;expires=" + objExpireDate.toGMTString();
});

var getCookie = (function (strName) {
	var strCookieName = strName + "=";
	var objCookie = document.cookie;
	
	if (objCookie.length > 0) 
	{
		var nBegin = objCookie.indexOf(strCookieName);
		
		if (nBegin < 0) 
		{
			return;
		}

		nBegin += strCookieName.length;
		
		var nEnd = objCookie.indexOf(";", nBegin);
		
		if (nEnd == -1) 
		{
			nEnd = objCookie.length;
		}
	}
	//alert(objCookie.substring(nBegin, nEnd));
	
	return unescape(objCookie.substring(nBegin, nEnd));
});	