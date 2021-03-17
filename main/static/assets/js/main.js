//MAIN
var mainSlider;
var mainInit = {
	init : function(node){
		this.node = node;
		this.len = node.find(" > ul > li").length;
		this.html = node.find(".playList").html();

		this.dotSet();
		this.product();

		if(this.len == 1){
			return;
		};
		
		var owner = this;
		var current = 0;
		var mainSlider = node.find(" > ul").bxSlider({
			controls : false,
			pager : false,
			auto : true,
			pause : 5000,
			onSlideBefore : function(ele, oldIndex, newIndex){
				owner.imgSet(newIndex);
			}
		});

		node.find(".left").on("click", function(e){
			e.preventDefault();
			mainSlider.goToPrevSlide();
		});

		node.find(".right").on("click", function(e){
			e.preventDefault();
			mainSlider.goToNextSlide();
		});

		node.find(".playList > li").on("click", function(e){
			e.preventDefault();

			mainSlider.goToSlide($(this).index());
		});
	},

	imgSet : function(chk){
		this.node.find(".playList > li").each(function(i){
			if(i == chk){
				$(this).addClass("on");
			}else{
				$(this).removeClass("on");
			};
		});
	},

	dotSet : function(chk){
		if(this.len == 1){
			this.node.find(".indicator, .left, .right").hide();
			return;
		};

		for(i = 1; i < this.len; i++){
			this.node.find(".playList").append(this.html);
		};

		this.node.find(".playList > li").each(function(i){
			if(i == 0){
				$(this).addClass("on");
			};

			$(this).find(".indent").text($(this).index() + 1);
		});
	},

	product : function(){
		//var linkArr = ["/kr/builders/index.do", "/kr/industrial/index.do", "/kr/ship/index.do", "/kr/car/index.do"];
		if (navigator.userAgent.match(/iPhone|iPod|iPad|Android|Windows CE|BlackBerry|Symbian|Windows Phone|webOS|Opera Mini|Opera Mobi|POLARIS|IEMobile|lgtelecom|nokia|SonyEricsson/i) != null ) {
			$(".main_product li").click(function(e) {
				e.preventDefault();
				$(this).find(" > a > img").eq(1).stop().fadeIn(500);

				$(this).find(".object").delay(500).stop().animate({right : 0}, 500, function(){
					location.href = $(this).parent().attr("href");
				});
			});
		}else{
			$(".main_product li").hover(function() {
				$(this).find(" > a > img").eq(1).stop().fadeIn(500);

				$(this).find(".object").delay(500).stop().animate({right : 0}, 500);
			}, function(){
				
				$(this).find(" > a > img").eq(1).stop().fadeOut(500);
				$(this).find(".object").stop().animate({right : "-11.6%"}, 500);
			});
		};
	}
};

$(document).ready(function(){
	
});