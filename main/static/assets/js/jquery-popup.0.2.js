jQuery.Popup = function(options){
    this._settings = {
        popupId : 'popup',
        popupTitle : '',
        width: 500,
        left: 0,
        top: 0,
        center: false,
        closeFlag : 'N',
        popupMove: true,
        popupBackground : true,
        popup_zindex : 1000,
        _popupBackgroundId : 'popup_bg',
        reload : false,
        popupHtml : '',
		lnkUrl : '',
		lnkTarget : ''
    };

    for (var i in options) {
        if (options.hasOwnProperty(i)){
            this._settings[i] = options[i];
        }
    }
};
jQuery.Popup.prototype = {
    openPopup : function(html, id, background){
        var id = this._settings.popupId;
        var width = this._settings.width;
        var left = this._settings.left;
        var top = this._settings.top;
        var center = this._settings.center;
        var closeFlag = this._settings.closeFlag;
        var bgBody = this._settings.popupBackground;
        var title = this._settings.popupTitle;
        var move = this._settings.popupMove;
        var zindex = this._settings.popup_zindex;
        var reload = this._settings.reload;
        var html = this._settings.popupHtml;
		var lnkUrl = this._settings.lnkUrl;
		var lnkTarget = this._settings.lnkTarget;
        var callback = this._settings.callback;
        
        if(bgBody) this._addBackground();
        if($("#"+id) && reload){
            $("#"+id).remove();
        }
		
		
        if($('#'+id).size() == 0){
			var inHtml = '<div id="'+id+'" class="layer_pop popDesign" style="z-index:'+zindex+';">';
				inHtml += '  <h1>' + title + '</h1>';
				inHtml += '  <div class="layer_cont">';
				inHtml += 	   html ;
				inHtml += '    <div class="btn_C">';
				inHtml += '	     <a href="#" onclick="javascript:fnPopClose(\''+id+'\'); return false;" class="btn_blue">닫기</a>';
				inHtml += '    </div>';
				inHtml += '    <a href="#" onclick="javascript:fnPopClose(\''+id+'\'); return false;" class="layer_close"><span class="indent">닫기</span></a>';
				inHtml += '  </div>';
				inHtml += '</div>';
								
        	//$(inHtml).appendTo(document.body);
			$("#layerPopup").append(inHtml);
           
        }	

        if(move) $("#"+id).draggable();
        var doc = document.documentElement, body = document.body;

        if(callback) callback();
    },

    closePopup : function(id){
        var id;
        if(!id){
            id = this._settings.popupId;
        }
        
        $("#"+id).remove();
        this._removeBackground();

    },

    _addBackground: function(){ 
        var body = document.body;
        var backgroundId = this._settings._popupBackgroundId;
        var zindex	= this._settings.popup_zindex - 1;
        if($('#'+backgroundId).size()>0){
            $('#'+backgroundId).addClass("layerpopup-background");
        }else {
            $('<div id="'+backgroundId+'" class="layerpopup-background" style="position:fixed;background-color:#000;opacity:0.6;z-index:' + zindex + ';top:0;left:0;"></div>').appendTo(document.body);
        } 
        $('#'+backgroundId).width(0);
        $('#'+backgroundId).height(0);
        $('#'+backgroundId).width(body.clientWidth);
        $('#'+backgroundId).height(body.clientHeight);
    },
    _removeBackground : function(){
    	
        var backgroundId = this._settings._popupBackgroundId;
        if($(".popup").size() < 1) {
            $('#'+backgroundId).removeClass("layerpopup-background");
            $('#'+backgroundId).remove();
        }
    } 
};