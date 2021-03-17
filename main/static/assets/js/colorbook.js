var colorInfoInit = function(config){
	this.config = {
		selector : config.selector || '',
		deviceChk : config.device || ''
	};
	
	this.init();
};

colorInfoInit.prototype = {
	init : function(){
		var owner = this;

		this.sel = this.config.selector;

		if(this.config.deviceChk == "M"){
			this.sel.addClass("mobile");
		};

		this.sel.find(".colorinfo_in .flip > a").on("click", function(e){
			e.preventDefault();
						
			if(parseInt(owner.sel.css("height")) == 57){
				/* 2018-03-20 베스트 컬러 추가 */
				var isBestColor = owner.sel.data("src");
				if(isBestColor == "best") {
					owner.sel.stop().animate({height : "540px"}, 500);
				} else {
					owner.sel.stop().animate({height : "260px"}, 500);
				}
				/* //2018-03-20 베스트 컬러 추가 */

				$(this).removeClass("on");
			}else{
				owner.sel.stop().animate({height : "57px"}, 500);
				$(this).addClass("on");
			};
			
		});

		this.sel.find(".recent_color ul > li .colorChip").on("click", function(e){
			e.preventDefault();			
			owner.sel.find(".recent_color ul > li").removeClass("on");
			$(this).parents('li').addClass("on");
		});

		this.scrollHn();
		//this.resizeHn();	
		
	},

	scrollHn : function(){
		var owner = this;
		
		$(window).on("scroll", function() {
			if($(window).width() < 960){

			}else{
				if($(window).scrollTop() + $(window).height() >= $(document).height() - $("#footer_wrap").height()) {
					var targetH = $(window).scrollTop() + $(window).height() - $("#footer_wrap").offset().top + 20;
					owner.sel.css("bottom", targetH + "px");
				}else{
					owner.sel.css("bottom", "20px");
				};
			};
			
		});
	},

	openLayer : function(){
		
		if(parseInt(this.sel.css("height")) == 57){
			var isBestColor = this.sel.data("src");

			if(isBestColor == "best") {
				this.sel.stop().animate({height : "540px"}, 500);
			}else{
				this.sel.stop().animate({height : "260px"}, 500);				
			}
			
			this.sel.find(".colorinfo_in .flip > a").removeClass("on");
		};
	},

	resizeHn : function(){
		if(this.config.deviceChk == "M"){
			var tT = ($(window).height() - this.sel.height()) / 2// + $(window).scrollTop();
			
			//this.sel.css({"top" : tT + "px", "bottom" : "auto", "height" : "auto"});
			//this.sel.css({"top" : "50%", "margin-top" : "-264px", "bottom" : "auto", "height" : "auto"});
			//$("#fade").hide();
			//alert(this.sel.height() + " :::::: " + $(window).scrollTop())
		}else{
			$("#fade").hide();
			
			var styleH = this.sel.height() + 2;
			this.sel.removeAttr("style");
			

			if($(window).scrollTop() + $(window).height() >= $(document).height() - $("#footer_wrap").height()) {
				var targetH = $(window).scrollTop() + $(window).height() - $("#footer_wrap").offset().top;
				this.sel.css({"bottom" : targetH + "px", "height" : styleH + "px"});
			}else{
				this.sel.css({"bottom" : 0, "height" : styleH + "px"});
			};
		};
	}
};

var colorInfo;

$(document).ready(function(){
	if (navigator.userAgent.match(/iPhone|iPod|iPad|Android|Windows CE|BlackBerry|Symbian|Windows Phone|webOS|Opera Mini|Opera Mobi|POLARIS|IEMobile|lgtelecom|nokia|SonyEricsson/i) != null ) {
		deviceChk = "M";
	}else{
		deviceChk = "W";
	}
	
	var colorInfoConfig = {
		selector : $(".colorinfo_layer"),
		device : deviceChk
	};
	colorInfo = new colorInfoInit(colorInfoConfig);
	
	//colorboard click
	$(".color_board > ul > li").on("click", function(e){
		e.preventDefault();
		if($(window).width() < 960){
			$(".colorinfo_layer").show();
			$("#fade").show();
			colorInfo.resizeHn();
		}else{
			colorInfo.openLayer();
		};
	});

	$(".colorinfo_layer .layer_close").on("click", function(e){
		e.preventDefault();
		$(".colorinfo_layer").hide();
		$("#fade").hide();
	});
				
});